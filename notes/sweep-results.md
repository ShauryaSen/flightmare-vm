# Sweep Results

**Conditions:** 12

## Ranked by eval_max_gate_idx → eval_mean_reward

| rank | condition | eval_gate | eval_rew | eval_len | lap? | final_rew | slope/M | min |
|------|-----------|-----------|----------|----------|------|-----------|---------|-----|
| 1 | `07_dev_ratio_high` | 3.0 | 24.70 | 167 | no | 11.13 | 1.68 | 42.3 |
| 2 | `04_gate_bonus_big` | 2.0 | 22.81 | 121 | no | 7.24 | 0.14 | 42.3 |
| 3 | `06_dev_coeff_low` | 2.0 | 19.56 | 150 | no | 11.32 | -0.09 | 42.3 |
| 4 | `02_harsh_crash` | 2.0 | 18.45 | 113 | no | 10.79 | 0.39 | 42.1 |
| 5 | `01_no_crash_penalty` | 2.0 | 17.23 | 136 | no | 10.32 | 1.99 | 41.8 |
| 6 | `11_combo_racing_line` | 2.0 | 16.21 | 143 | no | 15.74 | -0.99 | 66.0 |
| 7 | `05_dev_coeff_high` | 2.0 | 14.37 | 123 | no | 7.66 | -0.39 | 42.1 |
| 8 | `03_gate_bonus_on` | 2.0 | -4.74 | 135 | no | 9.74 | 0.87 | 42.1 |
| 9 | `08_ent_low` | 1.0 | 9.90 | 121 | no | 8.61 | 0.30 | 42.2 |
| 10 | `10_combo_positive_only` | 1.0 | 6.80 | 111 | no | 7.47 | 0.13 | 42.3 |
| 11 | `00_baseline` | 0.0 | -38.03 | 135 | no | 14.63 | 0.42 | 42.2 |
| 12 | `09_ent_high` | 0.0 | -250.07 | 133 | no | 9.86 | 0.49 | 42.4 |

## Interpretation

- **Tied top tier: ['07_dev_ratio_high', '04_gate_bonus_big', '06_dev_coeff_low']**. Top 3 are within ~1 gate of each other; could be seed noise. Re-run those with 3 seeds at 40M to break the tie.
- **Steep-slope dark horses (slope/M > 1.0):** ['07_dev_ratio_high', '01_no_crash_penalty']. These may not have converged — promote to the finalist re-run.
- **Likely-broken runs (ep_len_mean < 100):** ['07_dev_ratio_high', '04_gate_bonus_big', '06_dev_coeff_low', '02_harsh_crash', '01_no_crash_penalty', '11_combo_racing_line', '05_dev_coeff_high', '03_gate_bonus_on', '08_ent_low', '10_combo_positive_only', '00_baseline', '09_ent_high']. Drone is crashing fast — config may be reward-hacking or destabilizing.

## Decision rules used

1. Primary: `eval_max_gate_idx` (deterministic eval from start_pos)
2. Tiebreaker 1: `eval_lap_completed`
3. Tiebreaker 2: `eval_mean_reward`
4. Watch: `slope_last_2M` for steep-slope candidates that needed more training
5. Sanity: `ep_len_mean` (1500 = full episode, ~90 = crashing fast)
