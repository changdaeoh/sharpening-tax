# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""pass@k, pass^k, scalability and the Sharpening Tax from per-task success counts.

Per-task unbiased estimators (Chen et al., 2021) from c_i successes in n_i rollouts; A and S
are computed on the equal-weight task-mean curves:

    pass@k_i = 1 - C(n_i - c_i, k) / C(n_i, k)            coverage
    pass^k_i = C(c_i, k) / C(n_i, k)                      consistency
    A(K)     = sum_{k=1}^{K-1} [pass@K - pass@k]          raw area scalability
    S(K)     = A(K) / ((K - 1) (1 - pass@1))              calibrated scalability
    Tax_X(K) = X_base(K) - X_RL(K),   X in {A, S}         Sharpening Tax

Each policy uses its own full n_i; the tax is over the tasks both were evaluated on.
CLI:  python -m sharpening_tax.metrics --base base.jsonl --rl rl.jsonl [--K 128]
"""
from __future__ import annotations

import argparse

import numpy as np

from .outcomes import load_counts


def _grid(c, n, K):
    c, n = np.asarray(c, dtype=np.float64), np.asarray(n, dtype=np.float64)
    if K > n.min():
        raise ValueError(f"K={K} exceeds the smallest number of rollouts per task ({int(n.min())})")
    return c[:, None], n[:, None], np.arange(K)


def pass_at_k(c, n, K: int) -> np.ndarray:
    """[T, K] per-task unbiased pass@k, k = 1..K, via the stable product C(n-c, k) / C(n, k) = prod_{j<k} (n-c-j) / (n-j)."""
    c, n, j = _grid(c, n, K)
    return 1.0 - np.cumprod(np.clip(n - c - j, 0.0, None) / (n - j), axis=1)


def pass_hat_k(c, n, K: int) -> np.ndarray:
    """[T, K] per-task unbiased pass^k = C(c, k) / C(n, k), k = 1..K."""
    c, n, j = _grid(c, n, K)
    return np.cumprod(np.clip(c - j, 0.0, None) / (n - j), axis=1)


def scalability(curves: np.ndarray, K: int) -> tuple[np.ndarray, np.ndarray]:
    """A(K), S(K) of a curve [kmax] or a batch [R, kmax], curve[k-1] = pass@k; S is NaN if pass@1 = 1."""
    curves = np.atleast_2d(curves)
    A = (curves[:, K - 1: K] - curves[:, : K - 1]).sum(axis=1)
    headroom = 1.0 - curves[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        S = np.where(headroom > 0, A / headroom / (K - 1), np.nan)
    return A, S


class Pair:
    """A base / post-trained pair on one benchmark: per-task counts on the shared tasks."""

    def __init__(self, base: dict, rl: dict):
        self.tasks = sorted(set(base) & set(rl))
        if not self.tasks:
            raise ValueError("the two policies share no task_id")
        self.cb = np.array([base[t][0] for t in self.tasks])
        self.nb = np.array([base[t][1] for t in self.tasks])
        self.cr = np.array([rl[t][0] for t in self.tasks])
        self.nr = np.array([rl[t][1] for t in self.tasks])
        self.K = int(min(self.nb.min(), self.nr.min()))
        self.MB, self.MR = pass_at_k(self.cb, self.nb, self.K), pass_at_k(self.cr, self.nr, self.K)

    def coverage(self):
        return self.MB.mean(0), self.MR.mean(0)

    def consistency(self):
        return pass_hat_k(self.cb, self.nb, self.K).mean(0), pass_hat_k(self.cr, self.nr, self.K).mean(0)

    def tax(self, Ks, n_boot: int = 1000, seed: int = 0) -> dict:
        """{K: A and S of both arms, tax_A, tax_S, and *_ci95}. The CIs are 95% percentile intervals of a
        paired task bootstrap (one task draw shared by both arms and all K) of tax_A, tax_S and
        coverage_gap = pass@K base - RL; tax_S uses the draws where both S are defined, else [nan, nan]."""
        cov_b, cov_r = self.coverage()
        out = {}
        for K in Ks:
            (A_b,), (S_b,) = scalability(cov_b, K)
            (A_r,), (S_r,) = scalability(cov_r, K)
            out[K] = {"A_base": A_b, "A_rl": A_r, "S_base": S_b, "S_rl": S_r,
                      "tax_A": A_b - A_r, "tax_S": S_b - S_r}
        if n_boot:
            T = len(self.tasks)
            rng = np.random.default_rng(seed)
            W = np.stack([np.bincount(rng.integers(0, T, T), minlength=T) for _ in range(n_boot)]) / T
            boot_b, boot_r = W @ self.MB, W @ self.MR                  # [n_boot, K] curves
            for K in Ks:
                A_b, S_b = scalability(boot_b, K)
                A_r, S_r = scalability(boot_r, K)
                ok = np.isfinite(S_b) & np.isfinite(S_r)
                for name, v in (("tax_A", A_b - A_r), ("tax_S", (S_b - S_r)[ok]),
                                ("coverage_gap", boot_b[:, K - 1] - boot_r[:, K - 1])):
                    out[K][f"{name}_ci95"] = np.percentile(v, [2.5, 97.5]).tolist() if len(v) else [np.nan, np.nan]
        return out


def task_categories(c, n) -> dict:
    """Shares of always pass (c = n), pass given compute (0 < c < n), always fail (c = 0)."""
    c, n = np.asarray(c), np.asarray(n)
    return {"always_pass": float(np.mean(c == n)),
            "pass_given_compute": float(np.mean((c > 0) & (c < n))),
            "always_fail": float(np.mean(c == 0))}


def split_rollouts(c, n, n_est: int, rng: np.random.Generator, R: int, k_est: int,
                   k_val: int, k_all: int | None = None):
    """R random splits of each task's rollouts into n_est estimation and n - n_est validation rollouts;
    returns the mean pass@k curves of both halves ([R, k_est], [R, k_val]) and, with k_all, the
    validation mean pass^{k_all} ([R]). Rollouts are exchangeable and the estimators depend only on
    (c, n), so drawing c_est ~ Hypergeometric(c, n - c, n_est) equals partitioning the real rollouts."""
    c, n = np.asarray(c), np.asarray(n)
    T = len(c)
    c_est = rng.hypergeometric(c, n - c, n_est, size=(R, T))
    est, val = np.zeros((R, k_est)), np.zeros((R, k_val))
    alls = np.zeros(R)
    for nn in np.unique(n):                         # lookup tables per rollout count
        m = n == nn
        lut_e = pass_at_k(np.arange(n_est + 1), np.full(n_est + 1, n_est), k_est)
        lut_v = pass_at_k(np.arange(nn - n_est + 1), np.full(nn - n_est + 1, nn - n_est), k_val)
        c_val = (c[None, :] - c_est)[:, m]
        est += lut_e[c_est[:, m]].sum(axis=1)
        val += lut_v[c_val].sum(axis=1)
        if k_all:
            lut_a = pass_hat_k(np.arange(nn - n_est + 1), np.full(nn - n_est + 1, nn - n_est), k_all)[:, -1]
            alls += lut_a[c_val].sum(axis=1)
    return est / T, val / T, alls / T


def _ranks(v: np.ndarray) -> np.ndarray:
    """Ranks 0..len(v)-1, ties given their average rank."""
    r = np.empty(len(v))
    r[np.argsort(v, kind="mergesort")] = np.arange(len(v))
    _, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
    return (np.bincount(inv, r) / cnt)[inv]


def spearman(x, y) -> float:
    """Spearman rank correlation over the pairs where both values are finite."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    rx, ry = _ranks(x[m]), _ranks(y[m])
    rx -= rx.mean()
    ry -= ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description="pass@k / pass^k and the Sharpening Tax of one pair")
    ap.add_argument("--base", required=True, help="outcome JSONL of the base model")
    ap.add_argument("--rl", required=True, help="outcome JSONL of the post-trained model")
    ap.add_argument("--K", type=int, default=None, help="budget (default: the smallest number of rollouts per task in either file)")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    pair = Pair(load_counts(args.base), load_counts(args.rl))
    K = pair.K if args.K is None else args.K
    if not 2 <= K <= pair.K:
        ap.error(f"--K must be in [2, {pair.K}], the smallest number of rollouts per task of either model")
    cov_b, cov_r = pair.coverage()
    con_b, con_r = pair.consistency()
    print(f"{len(pair.tasks)} shared tasks\n{'k':>5} | {'pass@k base':>11} {'pass@k RL':>10} | "
          f"{'pass^k base':>11} {'pass^k RL':>10}")
    for k in [k for k in (1, 2, 4, 8, 16, 32, 64, 128, 256) if k < K] + [K]:
        print(f"{k:>5} | {cov_b[k-1]:>11.4f} {cov_r[k-1]:>10.4f} | {con_b[k-1]:>11.4f} {con_r[k-1]:>10.4f}")
    t = pair.tax([K], n_boot=args.n_boot)[K]
    print(f"\nTax_A({K}) = {t['tax_A']:+.3f}   (A: base {t['A_base']:.3f}, RL {t['A_rl']:.3f})"
          + (f"  95% CI {np.round(t['tax_A_ci95'], 3).tolist()}" if args.n_boot else ""))
    print(f"Tax_S({K}) = {t['tax_S']:+.4f}  (S: base {t['S_base']:.4f}, RL {t['S_rl']:.4f})"
          + (f"  95% CI {np.round(t['tax_S_ci95'], 4).tolist()}" if args.n_boot else ""))


if __name__ == "__main__":
    main()
