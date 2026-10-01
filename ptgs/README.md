# PTGS: posterior-tempered group sampling

Contents:

- `controller.py`: the tempering rule and the per-prompt Beta posterior;
- `integration.py`: `PTGSHooks`, the four places PTGS touches a group-based RL trainer, plus the
  corresponding edits for a verl-style trainer such as [RAGEN](https://github.com/mll-lab-nu/RAGEN).

`python -m ptgs` prints the tempering rule, the target ramp and one prompt's trajectory.
`examples/ptgs_toy.py` compares PTGS with fixed-temperature RL on a toy task.

## Usage

```python
from ptgs import PTGS

ptgs = PTGS(tau=1.5, gamma=0.95, pivot_start=0.25, pivot_end=0.5)   # an example setting
for step in range(1, total_steps + 1):
    ptgs.set_progress(step, total_steps)                            # target ramp 0.25 -> 0.5
    temps = {x: ptgs.temperature(x) for x in batch}                 # one T per prompt group
    groups = {x: policy.generate(x, n=16, temperature=temps[x]) for x in batch}
    for x, g in groups.items():
        ptgs.update(x, successes=num_success(g), n=16)              # every group, before any filter
    rl_update(groups, log_probs=policy.log_probs(groups, temperature=temps))   # at T_x
```

Three details matter:
- the task pool must be finite and cycled, so that each prompt's posterior accumulates;
- the posterior update must see every group before a reward-variance filter drops the all-fail
  ones;
- log-probs (and hence entropy) must be computed at each prompt's own sampling temperature.

`configs/ptgs_qwen2.5-7b.yaml` is an example RL configuration with PTGS (PPO, with a GRPO
variant).

## At inference time

PTGS is mainly a training-time sampler, but the same class can also temper a frozen policy.
For each prompt `x`, draw `m` pilot rollouts at `t_ref` and count their successes `s`, then

```python
c = PTGS(tau, t_ref=0.5, pivot_start=0.5, pivot_end=0.5)
c.update(x, successes=s, n=m)
T = c.temperature(x)          # sample the scored rollouts at this temperature
```
