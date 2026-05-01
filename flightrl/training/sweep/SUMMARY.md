# Sweep Results

**Conditions:** 1

## Ranked by eval_max_gate_idx → eval_mean_reward

| rank | condition | eval_gate | eval_rew | eval_len | lap? | final_rew | slope/M | min |
|------|-----------|-----------|----------|----------|------|-----------|---------|-----|
| 1 | `00_baseline` | 0.0 | -9.76 | 152 | no | -12.21 | 24.81 | 0.4 |

## Interpretation

- **No condition learned to clear gates reliably** (best max_gate=0). This suggests a deeper bug exists (reward shape, sim timing, init distribution). Hyperparameter sweep won't fix this — investigate before more training.
- **Steep-slope dark horses (slope/M > 1.0):** ['00_baseline']. These may not have converged — promote to the finalist re-run.
- **Likely-broken runs (ep_len_mean < 100):** ['00_baseline']. Drone is crashing fast — config may be reward-hacking or destabilizing.

## Decision rules used

1. Primary: `eval_max_gate_idx` (deterministic eval from start_pos)
2. Tiebreaker 1: `eval_lap_completed`
3. Tiebreaker 2: `eval_mean_reward`
4. Watch: `slope_last_2M` for steep-slope candidates that needed more training
5. Sanity: `ep_len_mean` (1500 = full episode, ~90 = crashing fast)
