# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Estimator correctness and the identities the metrics satisfy."""
from math import comb

import numpy as np

from sharpening_tax.metrics import Pair, pass_at_k, pass_hat_k, scalability, spearman, split_rollouts

RNG = np.random.default_rng(0)


def _counts(T=200, n=32, a=0.5, b=1.2):
    return RNG.binomial(n, RNG.beta(a, b, T)), np.full(T, n)


def _pair(base, rl):
    return Pair(dict(enumerate(base)), dict(enumerate(rl)))


def test_estimators_match_binomial_formulas():
    for n in (1, 5, 32, 128):
        for c in range(n + 1):
            cov, con = pass_at_k([c], [n], n)[0], pass_hat_k([c], [n], n)[0]
            for k in range(1, n + 1):
                assert np.isclose(cov[k - 1], 1 - comb(n - c, k) / comb(n, k))
                assert np.isclose(con[k - 1], comb(c, k) / comb(n, k))


def test_estimators_are_unbiased():
    """With c ~ Binomial(n, p), the estimators average to 1 - (1 - p)^k and p^k."""
    rng = np.random.default_rng(0)
    n, p, trials = 32, 0.3, 40_000
    c = rng.binomial(n, p, trials)
    cov, con = pass_at_k(c, np.full(trials, n), n).mean(0), pass_hat_k(c, np.full(trials, n), n).mean(0)
    for k in (1, 4, 16):
        assert abs(cov[k - 1] - (1 - (1 - p) ** k)) < 5e-3
        assert abs(con[k - 1] - p ** k) < 5e-3


def test_pass1_is_mean_success_rate():
    c, n = _counts()
    assert np.isclose(pass_at_k(c, n, 1).mean(), (c / n).mean())
    assert np.allclose(pass_at_k(c, n, 1), pass_hat_k(c, n, 1))


def test_fully_sharpened_policy_has_no_scalability():
    """Every task always or never solved: the pass@k curve is flat, so A = S = 0."""
    c, n = np.r_[np.full(5, 16), np.zeros(7, int)], np.full(12, 16)
    (A,), (S,) = scalability(pass_at_k(c, n, 16).mean(0), 16)
    assert A == 0 and S == 0


def test_s_is_nan_without_headroom():
    _, (S,) = scalability(pass_at_k([16] * 4, [16] * 4, 16).mean(0), 16)
    assert np.isnan(S)


def test_s_is_scale_free_in_the_number_of_tasks():
    """Duplicating the task set leaves the dataset-mean curve, and hence S, unchanged."""
    rng = np.random.default_rng(1)
    c, n = rng.binomial(32, rng.beta(0.5, 1.2, 200)), np.full(200, 32)
    _, (S,) = scalability(pass_at_k(c, n, 32).mean(0), 32)
    _, (S3,) = scalability(pass_at_k(np.tile(c, 3), np.tile(n, 3), 32).mean(0), 32)
    assert np.isclose(S, S3)


def test_area_equals_budget_times_ceiling_minus_average_coverage():
    """A(K) = K [C(K) - mean_{k<=K} pass@k]."""
    c, n = _counts()
    curve = pass_at_k(c, n, 32).mean(0)
    for K in (2, 8, 32):
        (A,), _ = scalability(curve, K)
        assert np.isclose(A, K * (curve[K - 1] - curve[:K].mean()))


def test_area_is_expected_number_of_redeemed_failures():
    """A(K) = E[(H - 1) 1{H <= K}], H = index of the first success."""
    p = RNG.beta(0.5, 1.5, 300)
    K = 16
    curve = np.mean([1 - (1 - p) ** k for k in range(1, K + 1)], axis=1)
    (A,), (S,) = scalability(curve, K)
    h = np.arange(1, K + 1)[:, None]
    pmf = (1 - p) ** (h - 1) * p                        # P(H = h) per task
    assert np.isclose(A, ((h - 1) * pmf).sum(0).mean())
    assert 0 <= S <= 1


def test_sharpening_charges_a_proportional_tax():
    """Sharpening a lam-fraction of tasks to p in {0, 1} gives Tax_A = lam * A_base."""
    p = np.repeat(RNG.beta(0.5, 1.5, 200), 2)          # each rate twice: sharpen one copy
    sharpened = np.tile([True, False], 200)
    p_rl = np.where(sharpened, (RNG.random(400) < 0.5).astype(float), p)
    K = 32
    area = lambda q: scalability(np.mean([1 - (1 - q) ** k for k in range(1, K + 1)], axis=1), K)[0][0]  # noqa: E731
    assert np.isclose(area(p) - area(p_rl), 0.5 * area(p))


def test_tax_is_zero_against_itself_and_signed():
    base = [(4, 16), (1, 16), (12, 16), (0, 16)]
    sharpened = [(16, 16), (0, 16), (16, 16), (0, 16)]
    t = _pair(base, base).tax([16], n_boot=0)[16]
    assert t["tax_A"] == 0 and t["tax_S"] == 0
    assert _pair(base, sharpened).tax([16], n_boot=0)[16]["tax_A"] > 0
    assert _pair(sharpened, base).tax([16], n_boot=0)[16]["tax_A"] < 0


def test_tax_and_bootstrap():
    cb, nb = _counts(a=0.6, b=1.0)
    cr, nr = np.where(RNG.random(len(cb)) < 0.5, np.where(cb > 0, nb, 0), cb), nb   # sharpened sibling
    pair = Pair({i: (int(c), int(n)) for i, (c, n) in enumerate(zip(cb, nb))},
                {i: (int(c), int(n)) for i, (c, n) in enumerate(zip(cr, nr))})
    t = pair.tax([8, 32], n_boot=200)
    for K in (8, 32):
        assert t[K]["tax_A"] > 0
        lo, hi = t[K]["tax_S_ci95"]
        assert lo <= t[K]["tax_S"] <= hi


def test_bootstrap_without_any_defined_s():
    """An arm that solves every task has no S; the other intervals are still reported."""
    t = _pair([(4, 4), (4, 4)], [(2, 4), (1, 4)]).tax([4], n_boot=20)[4]
    assert np.isnan(t["tax_S_ci95"]).all()
    assert np.isfinite(t["tax_A_ci95"]).all() and np.isfinite(t["coverage_gap_ci95"]).all()


def test_spearman_averages_ties():
    x, y = [0, 0, 0, 1, 2], [3, 2, 1, 4, 5]
    assert np.isclose(spearman(x, y), 2 / np.sqrt(5))    # as scipy.stats.spearmanr
    assert np.isclose(spearman(x[::-1], y[::-1]), spearman(x, y))


def test_rollout_splits_partition_successes():
    c, n = _counts(n=128)
    est, val, alls = split_rollouts(c, n, 16, np.random.default_rng(1), R=50, k_est=8, k_val=64, k_all=32)
    assert est.shape == (50, 8) and val.shape == (50, 64) and alls.shape == (50,)
    assert abs(est[:, 0].mean() - (c / n).mean()) < 0.01
    assert abs(val[:, 0].mean() - (c / n).mean()) < 0.01
