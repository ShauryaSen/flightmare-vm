#!/usr/bin/env python3
"""Plug-and-play track switcher for the flightmare gate-racing env.

Tracks live as standalone YAML files under flightlib/configs/tracks/. Each
holds a `start_pos` and a `gates:` list. This script swaps them into
flightlib/configs/quadrotor_env.yaml so the C++ env loads the chosen layout
on next startup. Reward / PPO / dynamics sections are left untouched.

Usage:
    python track_loader.py list
    python track_loader.py show <track>
    python track_loader.py apply <track>
    python track_loader.py current

Safety: refuses to apply while a sweep is running (detected by
sweep_overnight.py in the process list) so an in-flight condition isn't
clobbered. Override with --force if you really mean it.
"""

import argparse
import math
import subprocess
import sys
from pathlib import Path

from ruamel.yaml import YAML

THIS_DIR = Path(__file__).resolve().parent
FLIGHTMARE_ROOT = THIS_DIR.parent.parent
ENV_YAML = FLIGHTMARE_ROOT / "flightlib" / "configs" / "quadrotor_env.yaml"
TRACKS_DIR = FLIGHTMARE_ROOT / "flightlib" / "configs" / "tracks"

yaml = YAML()
yaml.preserve_quotes = True
yaml.indent(mapping=2, sequence=2, offset=0)


def load_yaml(path: Path):
    with open(path) as f:
        return yaml.load(f)


def dump_yaml(data, path: Path):
    with open(path, "w") as f:
        yaml.dump(data, f)


def gates_as_list(data):
    """Return gates as a Python list, or None if the file uses the legacy
    dict-of-named-gates format (incompatible with the current C++ loader)."""
    g = data.get("gates")
    if g is None:
        return []
    if isinstance(g, list):
        return list(g)
    return None  # legacy dict format


def list_tracks():
    tracks = sorted(TRACKS_DIR.glob("*.yaml"))
    if not tracks:
        print(f"No tracks found in {TRACKS_DIR}")
        return
    print(f"Available tracks ({TRACKS_DIR}):")
    print()
    for t in tracks:
        try:
            data = load_yaml(t)
            gates = gates_as_list(data)
            if gates is None:
                print(f"  {t.stem:<16}  -- legacy dict format, not applicable")
                continue
            desc = data.get("description", "")
            print(f"  {t.stem:<16} {len(gates):>2} gates   {desc}")
        except Exception as e:
            print(f"  {t.stem:<16} (failed to parse: {e})")


def track_arc_length(gates, start_pos):
    """Cheap straight-line sum start_pos → g0 → g1 → ... → gN. Approximate."""
    pts = [start_pos] + [g["position"] for g in gates]
    total = 0.0
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i - 1][0]
        dy = pts[i][1] - pts[i - 1][1]
        dz = pts[i][2] - pts[i - 1][2]
        total += math.sqrt(dx * dx + dy * dy + dz * dz)
    return total


def show_track(name):
    path = TRACKS_DIR / f"{name}.yaml"
    if not path.exists():
        sys.exit(f"Track not found: {path}")
    data = load_yaml(path)
    gates = gates_as_list(data)
    if gates is None:
        sys.exit(f"Track '{name}' uses the legacy dict-of-named-gates format and "
                 "is not compatible with the current env loader (which expects a "
                 "list under `gates:`). Convert it before applying.")
    sp = list(data.get("start_pos", [0, 0, 0]))
    print(f"Track:        {name}")
    print(f"Description:  {data.get('description', '(none)')}")
    print(f"Start pos:    [{sp[0]:.2f}, {sp[1]:.2f}, {sp[2]:.2f}]")
    print(f"Gates:        {len(gates)}")
    print(f"Approx arc:   {track_arc_length(gates, sp):.1f} m (straight-line sum)")
    print()
    print(f"  {'#':>2}  {'x':>8}  {'y':>8}  {'z':>6}   rotation")
    for i, g in enumerate(gates):
        p = g["position"]
        if "rotation_quat" in g:
            rq = g["rotation_quat"]
            rot = f"quat [{rq[0]:.3f}, {rq[1]:.3f}, {rq[2]:.3f}, {rq[3]:.3f}]"
        elif "rotation" in g:
            r = g["rotation"]
            rot = f"euler yaw={r[0]:.1f} pitch={r[1]:.1f} roll={r[2]:.1f}"
        else:
            rot = "(identity)"
        print(f"  {i:>2}  {p[0]:>8.2f}  {p[1]:>8.2f}  {p[2]:>6.2f}   {rot}")


def sweep_running():
    """Return True if sweep_overnight.py is in ps -e."""
    try:
        out = subprocess.check_output(["ps", "-e", "-o", "args"], text=True)
        for line in out.splitlines():
            if "sweep_overnight.py" in line and "track_loader" not in line:
                return True
    except Exception:
        pass
    return False


def current_track():
    """Best-effort: which track name matches the gates currently in env yaml?"""
    if not ENV_YAML.exists():
        return None, "env yaml missing"
    env = load_yaml(ENV_YAML)
    env_gates = gates_as_list(env) or []
    env_n = len(env_gates)
    env_first_pos = (
        tuple(round(c, 2) for c in env_gates[0]["position"]) if env_n else None
    )
    for t in sorted(TRACKS_DIR.glob("*.yaml")):
        data = load_yaml(t)
        gs = gates_as_list(data)
        if gs is None or len(gs) != env_n or env_n == 0:
            continue
        first = tuple(round(c, 2) for c in gs[0]["position"])
        if first == env_first_pos:
            return t.stem, None
    return None, f"no track matches (env has {env_n} gates)"


def apply_track(name, force=False):
    path = TRACKS_DIR / f"{name}.yaml"
    if not path.exists():
        sys.exit(f"Track not found: {path}\nRun `track_loader.py list` to see options.")

    if sweep_running() and not force:
        sys.exit(
            "REFUSED: sweep_overnight.py is running and would clobber the active "
            "condition. Wait for the sweep to finish, or pass --force if you "
            "really want to overwrite quadrotor_env.yaml right now."
        )

    track = load_yaml(path)
    track_gates = gates_as_list(track)
    if track_gates is None:
        sys.exit(f"Track '{name}' uses the legacy dict-of-named-gates format and "
                 "cannot be applied. Convert it to a list under `gates:` first.")
    if not track_gates or "start_pos" not in track:
        sys.exit(f"Track file missing gates or start_pos: {path}")

    env = load_yaml(ENV_YAML)
    env["start_pos"] = track["start_pos"]
    env["gates"] = track["gates"]
    dump_yaml(env, ENV_YAML)

    print(f"Applied track '{name}' to {ENV_YAML.relative_to(FLIGHTMARE_ROOT)}")
    print(f"  start_pos: {list(track['start_pos'])}")
    print(f"  gates:     {len(track['gates'])}")
    print()
    print("Next training run will pick this up automatically (the C++ env reads")
    print("the YAML at construction time).")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="List available tracks.")
    p_show = sub.add_parser("show", help="Show a track's gates.")
    p_show.add_argument("name")
    p_apply = sub.add_parser("apply", help="Patch quadrotor_env.yaml with a track.")
    p_apply.add_argument("name")
    p_apply.add_argument("--force", action="store_true",
                         help="Apply even if a sweep is running.")
    sub.add_parser("current", help="Best-effort: which track is currently loaded.")
    args = parser.parse_args()

    if args.cmd == "list":
        list_tracks()
    elif args.cmd == "show":
        show_track(args.name)
    elif args.cmd == "apply":
        apply_track(args.name, force=args.force)
    elif args.cmd == "current":
        name, err = current_track()
        if name:
            print(f"Current track: {name}")
        else:
            print(f"Current track: unknown ({err})")


if __name__ == "__main__":
    main()
