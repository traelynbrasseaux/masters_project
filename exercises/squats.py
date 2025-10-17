from typing import Dict, Tuple
import math
import json
from pathlib import Path

import cv2
import mediapipe as mp

from utils.geometry import calculate_angle, get_angle_color


class SquatsExercise:
    name = "squats"

    def __init__(self) -> None:
        self._rep_state = "up"
        self._correct_reps = 0
        self._incorrect_reps = 0

        # Angle zones (degrees)
        self.knee_safe = (90, 180)
        self.knee_caution = (80, 90)

        self.hip_safe = (160, 180)
        self.hip_caution = (140, 160)

        self.torso_safe = (160, 180)
        self.torso_caution = (140, 160)

        self._deg = "\u00B0"

        self._mp_pose = mp.solutions.pose

        # Optionally override with JSON config if present
        self._load_thresholds_from_config()

    # ---- Public API expected by pipeline ----
    def compute_metrics(self, landmarks, frame_shape: Tuple[int, int, int]) -> Dict[str, object]:
        h, w, _ = frame_shape

        def get_xy(idx: int) -> Tuple[int, int]:
            lm = landmarks[idx]
            return int(lm.x * w), int(lm.y * h)

        shoulder_l = get_xy(self._mp_pose.PoseLandmark.LEFT_SHOULDER)
        shoulder_r = get_xy(self._mp_pose.PoseLandmark.RIGHT_SHOULDER)
        hip = get_xy(self._mp_pose.PoseLandmark.LEFT_HIP)
        knee = get_xy(self._mp_pose.PoseLandmark.LEFT_KNEE)
        ankle = get_xy(self._mp_pose.PoseLandmark.LEFT_ANKLE)

        torso_center = ((shoulder_l[0] + shoulder_r[0]) // 2, (shoulder_l[1] + shoulder_r[1]) // 2)
        torso_ref = (torso_center[0], torso_center[1] - 50)

        knee_angle = calculate_angle(hip, knee, ankle)
        hip_angle = calculate_angle(torso_center, hip, knee)
        torso_angle = calculate_angle(torso_ref, torso_center, hip)

        return {
            "points": {
                "shoulder_l": shoulder_l,
                "shoulder_r": shoulder_r,
                "hip": hip,
                "knee": knee,
                "ankle": ankle,
                "torso_center": torso_center,
                "torso_ref": torso_ref,
            },
            "angles": {
                "knee": knee_angle,
                "hip": hip_angle,
                "torso": torso_angle,
            },
        }

    def classify_state(self, metrics: Dict[str, object]) -> Dict[str, object]:
        angles = metrics["angles"]  # type: ignore[index]
        knee = float(angles["knee"])  # type: ignore[index]
        hip = float(angles["hip"])  # type: ignore[index]
        torso = float(angles["torso"])  # type: ignore[index]

        knee_color = get_angle_color(knee, self.knee_safe, self.knee_caution)
        hip_color = get_angle_color(hip, self.hip_safe, self.hip_caution)
        torso_color = get_angle_color(torso, self.torso_safe, self.torso_caution)

        # Overall status: worst of the three
        def level(color):
            if color == (0, 255, 0):
                return 0
            if color == (0, 255, 255):
                return 1
            return 2

        worst = max(level(knee_color), level(hip_color), level(torso_color))
        status = "good" if worst == 0 else ("caution" if worst == 1 else "unsafe")
        reasons = []
        if status != "good":
            if level(knee_color) == worst:
                reasons.append("Knee angle out of range")
            if level(hip_color) == worst:
                reasons.append("Hip angle out of range")
            if level(torso_color) == worst:
                reasons.append("Torso alignment off")

        return {
            "status": status,
            "colors": {"knee": knee_color, "hip": hip_color, "torso": torso_color},
            "reasons": reasons,
        }

    def rep_update(self, metrics: Dict[str, object]) -> Dict[str, int]:
        knee_angle = float(metrics["angles"]["knee"])  # type: ignore[index]

        if knee_angle < 100 and self._rep_state == "up":
            self._rep_state = "down"
        elif knee_angle >= 160 and self._rep_state == "down":
            # Evaluate correctness at top of rep
            hip_angle = float(metrics["angles"]["hip"])  # type: ignore[index]
            torso_angle = float(metrics["angles"]["torso"])  # type: ignore[index]
            if (
                self.knee_safe[0] <= knee_angle <= self.knee_safe[1]
                and self.hip_safe[0] <= hip_angle <= self.hip_safe[1]
                and self.torso_safe[0] <= torso_angle <= self.torso_safe[1]
            ):
                self._correct_reps += 1
            else:
                self._incorrect_reps += 1
            self._rep_state = "up"

        return {"correct": self._correct_reps, "incorrect": self._incorrect_reps}

    def overlay(self, frame, metrics: Dict[str, object], classification: Dict[str, object], counts: Dict[str, int]) -> None:
        pts = metrics["points"]  # type: ignore[index]
        angles = metrics["angles"]  # type: ignore[index]
        colors = classification["colors"]  # type: ignore[index]

        knee = pts["knee"]  # type: ignore[index]
        hip = pts["hip"]  # type: ignore[index]
        ankle = pts["ankle"]  # type: ignore[index]
        torso_center = pts["torso_center"]  # type: ignore[index]
        torso_ref = pts["torso_ref"]  # type: ignore[index]

        # Draw joints and bones with thicker lines
        cv2.circle(frame, knee, 10, colors["knee"], -1)  # type: ignore[index]
        cv2.circle(frame, hip, 10, colors["hip"], -1)  # type: ignore[index]
        cv2.circle(frame, torso_center, 10, colors["torso"], -1)  # type: ignore[index]

        cv2.line(frame, hip, knee, colors["knee"], 6)  # type: ignore[index]
        cv2.line(frame, knee, ankle, colors["knee"], 6)  # type: ignore[index]
        cv2.line(frame, torso_center, hip, colors["torso"], 6)  # type: ignore[index]
        cv2.line(frame, torso_ref, torso_center, colors["torso"], 4)  # type: ignore[index]

        # Vertical guideline at torso x-position (faint)
        h_frame = frame.shape[0]
        cv2.line(frame, (torso_center[0], 0), (torso_center[0], h_frame), (60, 60, 60), 1)  # type: ignore[index]

        # Angle text
        cv2.putText(frame, f"Knee: {int(angles['knee'])}{self._deg}", (knee[0] - 60, knee[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors["knee"], 2)  # type: ignore[index]
        cv2.putText(frame, f"Hip: {int(angles['hip'])}{self._deg}", (hip[0] - 60, hip[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors["hip"], 2)  # type: ignore[index]
        cv2.putText(frame, f"Torso: {int(angles['torso'])}{self._deg}", (torso_center[0] - 60, torso_center[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors["torso"], 2)  # type: ignore[index]

        # Angle arcs at key joints for clearer visualization
        self._draw_angle_arc(frame, knee, hip, ankle, colors["knee"])  # type: ignore[index]
        self._draw_angle_arc(frame, hip, torso_center, knee, colors["hip"])  # type: ignore[index]
        self._draw_angle_arc(frame, torso_center, torso_ref, hip, colors["torso"])  # type: ignore[index]

        # HUD
        banner = classification["status"].upper()  # type: ignore[index]
        banner_color = (0, 200, 0) if banner == "GOOD" else ((0, 200, 200) if banner == "CAUTION" else (0, 0, 200))
        cv2.rectangle(frame, (20, 20), (380, 90), (30, 30, 30), -1)
        cv2.putText(frame, f"{banner}", (30, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.2, banner_color, 3)

        rep_text = f"Correct: {counts['correct']}  Incorrect: {counts['incorrect']}"
        cv2.putText(frame, rep_text, (400, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Reasons panel (if any)
        reasons = classification.get("reasons", [])  # type: ignore[assignment]
        if isinstance(reasons, list) and len(reasons) > 0:
            base_y = 100
            for i, reason in enumerate(reasons[:3]):
                y = base_y + i * 24
                cv2.putText(frame, f"- {reason}", (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, banner_color, 2)

    # ---- Internal helpers ----
    def _load_thresholds_from_config(self) -> None:
        """Load angle thresholds from configs/squats.json if available."""
        try:
            base = Path(__file__).resolve().parent.parent
            cfg_path = base / "configs" / "squats.json"
            if not cfg_path.exists():
                return
            with cfg_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            def to_tuple(pair):
                return (float(pair[0]), float(pair[1]))

            def to_ranges(value):
                if isinstance(value, list) and len(value) == 2 and not isinstance(value[0], list):
                    return to_tuple(value)
                if isinstance(value, list):
                    return [to_tuple(p) for p in value]
                return value

            if "knee_safe" in data:
                self.knee_safe = to_tuple(data["knee_safe"])  # type: ignore[assignment]
            if "knee_caution" in data:
                self.knee_caution = to_ranges(data["knee_caution"])  # type: ignore[assignment]

            if "hip_safe" in data:
                self.hip_safe = to_tuple(data["hip_safe"])  # type: ignore[assignment]
            if "hip_caution" in data:
                self.hip_caution = to_ranges(data["hip_caution"])  # type: ignore[assignment]

            if "torso_safe" in data:
                self.torso_safe = to_tuple(data["torso_safe"])  # type: ignore[assignment]
            if "torso_caution" in data:
                self.torso_caution = to_ranges(data["torso_caution"])  # type: ignore[assignment]
        except Exception:
            # Ignore config errors and keep defaults
            return

    def _draw_angle_arc(self, frame, center: Tuple[int, int], p1: Tuple[int, int], p2: Tuple[int, int], color: Tuple[int, int, int]) -> None:
        """Draw a small arc at 'center' spanning the smaller angle between center->p1 and center->p2."""
        cx, cy = center
        v1x, v1y = p1[0] - cx, p1[1] - cy
        v2x, v2y = p2[0] - cx, p2[1] - cy
        if v1x == 0 and v1y == 0:
            return
        if v2x == 0 and v2y == 0:
            return
        a1 = math.degrees(math.atan2(-v1y, v1x))
        a2 = math.degrees(math.atan2(-v2y, v2x))
        a1 = (a1 + 360.0) % 360.0
        a2 = (a2 + 360.0) % 360.0
        sweep = (a2 - a1) % 360.0
        if sweep > 180.0:
            a1, a2 = a2, a1
            sweep = (a2 - a1) % 360.0
        radius = 40
        cv2.ellipse(frame, (int(cx), int(cy)), (radius, radius), 0, a1, a1 + sweep, color, 3)


