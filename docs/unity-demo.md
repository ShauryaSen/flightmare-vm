# Unity demo of circle_8 trained policy

Live render from the VM-trained PPO policy → Unity Editor on Mac → screen-record.

## One-time SSH tunnel from Mac

The C++ env on the VM publishes drone state on TCP ports 10253 / 10254.
Unity on Mac needs to reach those. Open a terminal on Mac and run:

```bash
ssh -L 10253:localhost:10253 -L 10254:localhost:10254 ssen@<VM_IP_OR_HOST>
```

Leave that SSH session open. The forwarding is active for as long as it lives.

## On the VM (in the SSH session)

```bash
cd ~/flightmare/flightrl/training
python track_loader.py apply circle_8
source ~/miniforge3/etc/profile.d/conda.sh && conda activate flightmare
export FLIGHTMARE_PATH=~/flightmare

# Then START THE SCRIPT — but DO NOT hit play in Unity yet. The script will
# wait at "Trying to Connect Unity" until Unity is up and connected.
python viz_unity.py \
  --model saved/curriculum_v2/E_circle_8_v4/best_model.zip \
  --vecnorm saved/curriculum_v2/E_circle_8_v4/vec_normalize.pkl \
  --scene-id 1 \
  -n 5
```

`--scene-id` options: `0`=Industrial, `1`=Warehouse, `2`=Garage, `3`=Forest, `4`=Tunnels.
Warehouse (1) is usually cleanest for racing demos.

## On Mac (Unity Editor)

1. Open the `flightmare_unity` project (already open in your screenshot).
2. Open `Top_Level_Scene` if it isn't already.
3. **Hit Play** (the ▶ button at the top).
4. In the Game window:
   - IP field: leave as `127.0.0.1` (the SSH tunnel forwards to the VM).
   - Click **CONNECT**.
   - Click the scene button matching `--scene-id` you passed (default Warehouse).
5. The drone should appear flying through the gates.

When the script finishes 5 episodes it disconnects and Unity stops receiving updates.

## Recording the GIF / video on Mac

QuickTime is the fastest path:
1. **Cmd+Shift+5** → "Record Selected Portion" → drag a box around Unity's Game window → click Record.
2. Re-launch the python script on the VM (steps above). The drone runs again.
3. Stop recording when the run ends.
4. Save as `.mov` from QuickTime.
5. To convert to GIF: `ffmpeg -i recording.mov -vf "fps=30,scale=800:-1" -loop 0 demo.gif`
   (install ffmpeg via `brew install ffmpeg` if needed).

## Troubleshooting

- **"Unity Connection time out"** in the python output — Unity is not running or didn't click CONNECT.
- **Drone doesn't appear** — wrong scene_id (script publishes for whatever scene Unity loaded).
- **Video looks too fast** — the `--realtime` flag is on by default; if it's still fast, the env is dropping frames. Try `--n-episodes 3` for shorter clean runs.
- **vec_env.yaml looks edited after running** — script restores the original via `vec_env.yaml.viz_backup`. If something crashed mid-run, restore manually with `mv ~/flightmare/flightlib/configs/vec_env.yaml.viz_backup ~/flightmare/flightlib/configs/vec_env.yaml`.
