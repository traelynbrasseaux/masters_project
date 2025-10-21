# masters_project – Pose-based Exercise Form and Injury-Prevention Toolkit

Computer-vision toolkit for analyzing exercise form using pose estimation. It uses MediaPipe Pose and OpenCV to compute joint angles, classify form quality (good/caution/unsafe), overlay visual feedback, and count repetitions. The included exercise is `squats` with configurable angle thresholds.

## Project structure
```
configs/
  squats.json            # Angle thresholds for squats (safe/caution)
exercises/
  __init__.py
  squats.py              # Squats logic: metrics, classification, overlay, rep counting
utils/
  geometry.py            # Angle math, EMA smoothing, color mapping
pose_pipeline.py         # Webcam pipeline: capture → pose → metrics → overlay
registry.py              # Exercise registry and factory
web/
  index.html, app.js     # Optional browser demo using Tasks Vision (WASM)
requirements.txt         # Python dependencies
```

## Requirements
- Python 3.9–3.12 on Windows (PowerShell)
- A webcam

## Installation (Windows PowerShell)
Run these commands from the project root:

```powershell
# 1) Create a virtual environment
python -m venv .venv

# 2) Upgrade packaging tools inside the venv and install deps
.\.venv\Scripts\python -m pip install -U pip setuptools wheel
.\.venv\Scripts\python -m pip install -r requirements.txt
```

You can also activate the venv if preferred: `.\.venv\Scripts\Activate` (PowerShell), then use `python` directly.

## Running the desktop pipeline
```powershell
# one line
python .\pose_pipeline.py --exercise squats --proc-width 640 --proc-height 360 --display-scale 1.5 --model-complexity 0 --alpha 0.4

# or using the venv's python without activation
.\.venv\Scripts\python .\pose_pipeline.py --exercise squats --proc-width 640 --proc-height 360 --display-scale 1.5 --model-complexity 0 --alpha 0.4

# multiline in PowerShell (use backticks ` for line continuation)
python .\pose_pipeline.py --exercise squats `
  --proc-width 640 --proc-height 360 `
  --display-scale 1.5 `
  --model-complexity 0 `
  --alpha 0.4
```

Notes:
- Press `q` to quit the window.
- Flags: `--proc-width/--proc-height` control processing resolution, independent of display size.
- Use `--display-scale` to scale up the window (e.g., 1.5, 2.0) or `--fullscreen` to fill the screen.
- `--frame-skip N` processes every N+1 frames; higher values can improve performance.
- Angle zones can be customized in `configs/squats.json`.

## Optional: Web demo
Open `web/index.html` with a static server (e.g., VS Code Live Server or `python -m http.server` in the `web` folder) and allow camera access. The browser demo reproduces angle visualization and status HUD.
