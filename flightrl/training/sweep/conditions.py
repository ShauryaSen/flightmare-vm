"""Sweep condition definitions.

Each condition is a dict of YAML overrides applied on top of the locked
baseline (whatever quadrotor_env.yaml was at sweep start). Keys map to
quadrotor_env.yaml's `rl` section unless prefixed with `ppo.`, in which
case they override PPO hyperparameters in _train_one.py.
"""

CONDITIONS = [
    # ── Control ──────────────────────────────────────────────────────────
    {"name": "00_baseline", "rl": {}, "ppo": {}},

    # ── 1-factor: crash penalty ──────────────────────────────────────────
    {"name": "01_no_crash_penalty",   "rl": {"crash_penalty": 0.0},   "ppo": {}},
    {"name": "02_harsh_crash",        "rl": {"crash_penalty": -20.0}, "ppo": {}},

    # ── 1-factor: gate bonus ─────────────────────────────────────────────
    {"name": "03_gate_bonus_on",      "rl": {"gate_bonus": 5.0},      "ppo": {}},
    {"name": "04_gate_bonus_big",     "rl": {"gate_bonus": 20.0},     "ppo": {}},

    # ── 1-factor: deviation shaping ──────────────────────────────────────
    {"name": "05_dev_coeff_high",     "rl": {"deviation_penalty_coeff": 0.1},   "ppo": {}},
    {"name": "06_dev_coeff_low",      "rl": {"deviation_penalty_coeff": 0.001}, "ppo": {}},
    {"name": "07_dev_ratio_high",     "rl": {"spline_deviation_ratio": 3.0},    "ppo": {}},

    # ── 1-factor: PPO entropy ────────────────────────────────────────────
    {"name": "08_ent_low",  "rl": {}, "ppo": {"ent_coef": 0.001}},
    {"name": "09_ent_high", "rl": {}, "ppo": {"ent_coef": 0.05}},

    # ── Combos: high-value pairs ─────────────────────────────────────────
    {"name": "10_combo_positive_only",
     "rl": {"crash_penalty": 0.0, "gate_bonus": 5.0}, "ppo": {}},

    {"name": "11_combo_racing_line",
     "rl": {"deviation_penalty_coeff": 0.001}, "ppo": {"ent_coef": 0.05}},

    {"name": "12_combo_strict_racer",
     "rl": {"gate_bonus": 5.0, "deviation_penalty_coeff": 0.1}, "ppo": {}},
]
