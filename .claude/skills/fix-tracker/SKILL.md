---
name: fix-tracker
description: Use this skill whenever making code/config changes intended to fix the "drone stuck before first gate" problem (or any other ongoing debugging effort in this repo). Read FIXES_ATTEMPTED.md FIRST so you know what has already been tried, then append any new attempt with its hypothesis and expected effect. Triggers on any edit that is a debugging hypothesis rather than a feature/refactor.
---

# Fix Tracker

This repo has an ongoing debugging effort (drone racing RL, currently: "drone gets stuck before passing the first gate"). To avoid re-trying fixes that already failed and to keep a clear paper trail for ablations, every attempted fix is logged in [FIXES_ATTEMPTED.md](../../../FIXES_ATTEMPTED.md) at the repo root.

## When to use

Use this skill **before** making any change that is a hypothesis about a bug. Signals:
- Changing reward coefficients, dynamics parameters, observation logic, training hyperparams
- Adding/removing reward terms or termination conditions
- Tweaking initialization, normalization, or curriculum
- Anything where you'd say "I think this will fix X"

Do NOT use for: pure refactors, doc updates, adding tests, formatting, fixing build errors that are unrelated to the training problem.

## Workflow

1. **Read** [FIXES_ATTEMPTED.md](../../../FIXES_ATTEMPTED.md) before proposing any fix. Check whether the user has already tried this approach (and whether it worked, didn't work, or wasn't conclusive).
2. **Apply** the fix.
3. **Append** a new entry at the bottom of `FIXES_ATTEMPTED.md` using the template below. One entry per logical fix (a multi-file change for a single hypothesis = one entry).
4. If the user later reports whether the fix worked, **update** the entry's `Outcome:` line. Don't delete failed entries — failed attempts are the most valuable signal.

## Entry template

```markdown
## YYYY-MM-DD — <short title>

**Hypothesis:** <one or two sentences on what you think is wrong and why this change should help>

**Files changed:**
- `path/to/file.cpp:line` — <what changed>

**Expected effect:** <concrete, falsifiable prediction — what should the metric/behavior look like if this works?>

**Outcome:** _pending — to be filled in after user tests_
```

## Rules

- **One hypothesis per entry.** If you bundle five unrelated reward changes into one entry, you can't tell which one worked.
- **Be specific in "Expected effect."** "Drone should fly better" is useless. "ep_rew_mean should exceed 50 within 500K steps and gate-pass count should be > 0" is useful.
- **Don't edit old entries** except to update `Outcome:`. The log is append-only history.
- **Cite the source** (paper section, github issue, conversation) when the fix is based on external evidence.
