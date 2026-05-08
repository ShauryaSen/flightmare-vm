"""Adaptive complexity controller for Condition C — IROS21 §III.D.

Paper signal: "We use the crash ratio computed across all parallel
environments to adapt the complexity of the sampled tracks." Concretely,
we use the *completion ratio* — fraction of episodes that pass every gate
on the track at least once — which is the natural inverse of crash ratio
in our env (crash = ground impact, OOB, or timeout before completing the
track; gate-collision termination is disabled per Bug 8).

Schedule (additive steps so c=0 isn't a multiplicative trap):
  - c_init = 0.0  (straight-line baseline)
  - Every `update_every` completed episodes, compute completion_ratio
    over the rolling window.
    completion_ratio > hi  → c += step_up    (cap at 1.0)
    completion_ratio < lo  → c -= step_down  (floor at 0.0)
    in between           → hold
"""

from collections import deque


class AdaptiveComplexity:
    def __init__(self,
                 num_gates: int,
                 c_init: float = 0.0,
                 c_max: float = 1.0,
                 c_min: float = 0.0,
                 step_up: float = 0.05,
                 step_down: float = 0.05,
                 window: int = 100,
                 update_every: int = 100,
                 hi_completion: float = 0.70,
                 lo_completion: float = 0.30):
        self.num_gates = int(num_gates)
        self.c = float(c_init)
        self.c_max = float(c_max)
        self.c_min = float(c_min)
        self.step_up = float(step_up)
        self.step_down = float(step_down)
        self.window = int(window)
        self.update_every = int(update_every)
        self.hi = float(hi_completion)
        self.lo = float(lo_completion)

        self.recent_complete = deque(maxlen=self.window)  # bool per episode
        self.recent_gate_idx = deque(maxlen=self.window)  # for diagnostics
        self.episodes_since_update = 0
        self.total_episodes = 0
        self.history = []

    @property
    def complexity(self) -> float:
        return float(self.c)

    def push_episode(self, gates_passed: float, current_step: int = 0):
        """Record one finished episode. `gates_passed` is the cumulative count
        from the C++ env's extra_info — does not wrap (so >= num_gates means
        the drone completed at least one full lap). Returns True iff c moved.
        """
        completed = bool(gates_passed >= self.num_gates)
        self.recent_complete.append(completed)
        self.recent_gate_idx.append(float(gates_passed))
        self.episodes_since_update += 1
        self.total_episodes += 1

        if self.episodes_since_update < self.update_every:
            return False
        if len(self.recent_complete) < max(10, self.window // 2):
            self.episodes_since_update = 0
            return False

        completion_ratio = sum(self.recent_complete) / len(self.recent_complete)
        mean_gates = sum(self.recent_gate_idx) / len(self.recent_gate_idx)
        old_c = self.c
        if completion_ratio > self.hi:
            self.c = min(self.c + self.step_up, self.c_max)
        elif completion_ratio < self.lo:
            self.c = max(self.c - self.step_down, self.c_min)

        self.episodes_since_update = 0
        changed = abs(self.c - old_c) > 1e-9
        self.history.append({
            "step": int(current_step),
            "total_episodes": self.total_episodes,
            "completion_ratio": float(completion_ratio),
            "mean_gates": float(mean_gates),
            "c_before": float(old_c),
            "c_after": float(self.c),
        })
        return changed
