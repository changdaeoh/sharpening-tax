# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""`python -m ptgs`: print the tempering rule, the pivot ramp and one prompt's trajectory."""

from .controller import PTGS, pivot_at, temperature_from_p

tau, t_ref = 1.5, 1.0
print(f"Tempering rule, tau={tau}, T_ref={t_ref}\n")
print(f"{'p_hat':>7} |" + "".join(f"{f'p~={pv}':>10}" for pv in (0.25, 0.5)))
print("-" * 29)
for p in (0.0, 0.1, 0.25, 0.4, 0.5, 0.75, 1.0):
    row = "".join(f"{temperature_from_p(p, pv, tau, t_ref):10.3f}" for pv in (0.25, 0.5))
    print(f"{p:7.2f} |{row}")

print("\nPivot ramp over a 200-step run (geometric, 0.25 -> 0.50)")
print("  " + "  ".join(f"step {s:>3}: {pivot_at(s, 200):.3f}" for s in (1, 50, 100, 150, 200)))

print("\nOne prompt's trajectory, revisited every 25 steps: fails everything, then learns")
print("(n=16/group, gamma=0.95; each T is a single Thompson draw)")
ptgs, key = PTGS(), "task_0017"
for visit, observed_successes in enumerate([0, 0, 1, 3, 8, 13, 15, 16]):
    step = 1 + 25 * visit
    ptgs.set_progress(step, 200)
    mean = ptgs.posterior_mean(key)
    t = ptgs.temperature(key)
    print(
        f"  step {step:>3}  pivot={ptgs.pivot:.3f}  posterior mean={mean:.3f}  "
        f"-> T={t:.3f}   (observed {observed_successes:>2}/16 successes)"
    )
    ptgs.update(key, observed_successes, 16)
