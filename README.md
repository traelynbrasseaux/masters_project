# HealthTech Challenge – Pose-based Exercise Form and Injury-Prevention Toolkit

Computer-vision toolkit for analyzing exercise form using pose estimation. It uses MediaPipe Pose and OpenCV to compute joint angles, classify form quality (good/caution/unsafe), and overlay visual feedback. The included exercise is `squats` with configurable angle thresholds.

## Project structure
```
configs/
  squats.json            # Angle thresholds for squats (safe/caution)
exercises/
  __init__.py
  squats.py              # Squats logic: metrics, classification, overlay
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

## HealthTech Challenge alignment (ACL prevention and recovery)

This project targets ACL injury prevention and rehab support for adolescent and young adult athletes in Louisiana by delivering real‑time, pose‑based movement analysis, configurable safety thresholds, and explainable feedback that coaches and athletes can act on during training sessions.

### Evidence from the provided NCBI articles
- **ACL burden in football and return-to-play guidance**: A recent review synthesizing football‑specific ACL management reports NCAA football ACL injury rates of 14.4–18.0 per 100,000 athlete exposures; emphasizes objective, milestone‑based rehab; and notes a trend toward more deliberate return‑to‑play at ≥7–9 months to reduce reinjury risk. It also summarizes multicenter protocol efforts (e.g., MOON) and highlights the importance of strength, confidence, and football‑specific functional testing for clearance. [Link](https://pmc.ncbi.nlm.nih.gov/articles/PMC12014870/#:~:text=In%20their%20high%20school%20athlete,across%20all%20levels%20of%20competition)

- **Risk and prevention insights (systematic review)**: A high‑level synthesis of Level I/II studies (PMID: 38692337) reports higher ACL injury risk among female youth athletes and supports neuromuscular/technique training to address modifiable risk factors (e.g., dynamic valgus, landing mechanics, and trunk control). These insights inform our emphasis on movement quality metrics and coachable feedback. [Link](https://pubmed.ncbi.nlm.nih.gov/38692337/#:~:text=Results:%20A%20total%20of%201%2C389,Level%20I%20and%20II%20studies)

- **Adolescent ACL management and rehab progression**: An open‑access review (PMCID: PMC11439179) underscores milestone‑based progression, objective strength/functional testing, and patient‑reported outcomes when guiding return to sport in youth, aligning with our plan to surface objective readiness metrics and to avoid purely time‑based decisions. [Link](https://pmc.ncbi.nlm.nih.gov/articles/PMC11439179/?utm_source=chatgpt.com)

### How this toolkit addresses the problem
- **Wearables/real‑time biomechanics monitoring (computer vision)**: Uses webcam/pose estimation to approximate lower‑extremity kinematics, compute joint angles, and highlight risky patterns (e.g., excessive knee valgus, limited hip/knee flexion on descent, trunk sway) during common training tasks like squats and (extendable) jump‑landing and cutting drills.
- **Collaborative, data‑driven training**: Produces interpretable metrics and pass/fail thresholds (configurable in `configs/`) that coaches can standardize across teams; supports consistent technique cues derived from evidence‑based prevention programs.
- **AI‑driven risk and personalized rehab support**: The classification pipeline and thresholds can be tailored per athlete profile (sex, age group, training phase) and extended with progression gates that mirror milestone‑based clearance criteria (strength symmetry, functional tests) emphasized in the literature.
- **Telehealth and rural access**: Runs on commodity webcams with no dedicated sensors, enabling remote check‑ins and at‑home exercise supervision; outputs can be shared asynchronously with clinicians when specialty access is limited.

### What is supported today
- Pose pipeline with visual overlays and traffic‑light feedback for squats.
- Configurable angle thresholds per exercise in `configs/` for rapid tuning to local programs.
- Modular `exercises/` design for adding ACL‑relevant tasks (e.g., double‑/single‑leg squat, drop jump, landing mechanics, step‑down) that target dynamic valgus and trunk control.

### Planned extensions (for Louisiana context)
- Add jump‑landing and change‑of‑direction screens with knee valgus indexing and side‑to‑side comparisons.
- Simple readiness widgets (e.g., progression gates) to support milestone‑based rehab rather than time‑only decisions.
- Profile‑aware presets for female athletes and youth age bands consistent with prevention literature. (set on user side)

## References
- Rund JM, Christensen GV, Fleming JA, Wolf BR. Anterior Cruciate Ligament Tears among Football Players. Curr Rev Musculoskelet Med. 2025;18(5):183–189. [NCBI/PMC link](https://pmc.ncbi.nlm.nih.gov/articles/PMC12014870/#:~:text=In%20their%20high%20school%20athlete,across%20all%20levels%20of%20competition)
- PubMed (PMID: 38692337). Systematic review/meta‑analysis on ACL injury risk and prevention in youth/young athletes; higher risk in female athletes; supports neuromuscular/technique training. [PubMed link](https://pubmed.ncbi.nlm.nih.gov/38692337/#:~:text=Results:%20A%20total%20of%201%2C389,Level%20I%20and%20II%20studies)
- NCBI/PMC (PMCID: PMC11439179). Recent review on adolescent ACL management, milestone‑based rehab, and return‑to‑sport decision‑making. [NCBI/PMC link](https://pmc.ncbi.nlm.nih.gov/articles/PMC11439179/?utm_source=chatgpt.com)
