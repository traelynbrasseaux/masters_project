import argparse
from typing import Optional

import cv2
import mediapipe as mp

from registry import get_exercise, available_exercises
from utils.geometry import EMA


def parse_args():
    p = argparse.ArgumentParser(description="Pose-based form analysis")
    p.add_argument("--exercise", type=str, default="squats", help=f"Exercise to run. Options: {available_exercises()}")
    p.add_argument("--width", type=int, default=640, help="Capture width (camera request; may be ignored by driver)")
    p.add_argument("--height", type=int, default=360, help="Capture height (camera request; may be ignored by driver)")
    p.add_argument("--frame-skip", type=int, default=0, help="Process every N+1 frames; 0 = process every frame")
    p.add_argument("--model-complexity", type=int, default=0, choices=[0, 1, 2], help="MediaPipe Pose model complexity")
    p.add_argument("--alpha", type=float, default=0.4, help="EMA smoothing factor (0..1)")
    # New: independent processing size and display controls
    p.add_argument("--proc-width", type=int, default=640, help="Processing width used for pose + overlay")
    p.add_argument("--proc-height", type=int, default=360, help="Processing height used for pose + overlay")
    p.add_argument("--display-scale", type=float, default=1.0, help="Scale factor for the display window (e.g., 1.5, 2.0)")
    p.add_argument("--fullscreen", action="store_true", help="Open display window in fullscreen mode")
    return p.parse_args()


def run():
    args = parse_args()

    # Performance and capture configuration
    cv2.ocl.setUseOpenCL(False)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    exercise = get_exercise(args.exercise)
    ema = EMA(alpha=args.alpha)

    mp_pose = mp.solutions.pose

    with mp_pose.Pose(
        model_complexity=args.model_complexity,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        smooth_landmarks=True,
    ) as pose:
        frame_index = 0
        last_metrics: Optional[dict] = None
        last_classification: Optional[dict] = None
        last_counts: Optional[dict] = {"correct": 0, "incorrect": 0}
        window_name = f"Form Analysis - {exercise.name}"
        window_initialized = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            process_this = (args.frame_skip <= 0) or (frame_index % (args.frame_skip + 1) == 0)

            # Always resize to the processing size for consistent coordinates and overlay
            proc_w, proc_h = int(args.proc_width), int(args.proc_height)
            frame_proc = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR)

            if process_this:
                # Prepare for MediaPipe processing on the resized frame
                frame_rgb = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2RGB)
                frame_rgb.flags.writeable = False
                results = pose.process(frame_rgb)

                if results.pose_landmarks:
                    lm = results.pose_landmarks.landmark
                    metrics = exercise.compute_metrics(lm, frame_proc.shape)

                    # Smooth numeric metrics (angles)
                    angles = metrics.get("angles", {})
                    if isinstance(angles, dict):
                        for key, value in list(angles.items()):
                            if isinstance(value, (int, float)):
                                angles[key] = float(ema.apply(key, float(value)))

                    classification = exercise.classify_state(metrics)
                    counts = exercise.rep_update(metrics, classification)

                    last_metrics = metrics
                    last_classification = classification
                    last_counts = counts
                else:
                    # No pose detected; keep last state and show hint
                    last_metrics = None
                    last_classification = {"status": "no-pose", "colors": {}, "reasons": ["No pose detected"]}
            # Draw overlay using last known state on the resized processing frame
            draw = frame_proc.copy()
            if last_metrics and last_classification and last_counts:
                exercise.overlay(draw, last_metrics, last_classification, last_counts)
            else:
                cv2.putText(draw, "No pose detected", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50, 50, 255), 2)

            # Prepare display frame (scaled) and window properties
            scale = max(0.1, float(args.display_scale))
            if scale != 1.0:
                disp_w, disp_h = int(proc_w * scale), int(proc_h * scale)
                display_frame = cv2.resize(draw, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
            else:
                display_frame = draw

            if not window_initialized:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                if args.fullscreen:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    h_win, w_win = display_frame.shape[0], display_frame.shape[1]
                    try:
                        cv2.resizeWindow(window_name, w_win, h_win)
                    except Exception:
                        pass
                window_initialized = True

            cv2.imshow(window_name, display_frame)
            frame_index += 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()


