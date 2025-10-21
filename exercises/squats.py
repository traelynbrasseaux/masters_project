from typing import Dict, Tuple, Optional
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
        self._had_unsafe_in_rep = False

        # Angle zones (degrees)
        self.knee_safe = (90, 180)
        self.knee_caution = (80, 90)

        self.hip_safe = (160, 180)
        self.hip_caution = (140, 160)

        self.torso_safe = (160, 180)
        self.torso_caution = (140, 160)

        # Knee valgus thresholds (normalized offset relative to hip-ankle length)
        # Evaluate only when knee is flexed (angle < 150 deg) to avoid false positives at top
        self.valgus_caution_norm = 0.05  # 5%
        self.valgus_unsafe_norm = 0.10   # 10%

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

        # Knee valgus proxy: normalized signed offset of knee from the hip-ankle line.
        # Compute projection of knee onto the hip-ankle line and signed offset towards torso (medial) side.
        hx, hy = hip
        ax, ay = ankle
        kx, ky = knee
        vx, vy = ax - hx, ay - hy
        v_len_sq = float(vx * vx + vy * vy)
        if v_len_sq <= 1e-6:
            t = 0.0
            px, py = hx, hy
            norm_offset = 0.0
            inward = False
        else:
            wx, wy = kx - hx, ky - hy
            t = max(0.0, min(1.0, (wx * vx + wy * vy) / v_len_sq))
            px = hx + int(round(t * vx))
            py = hy + int(round(t * vy))
            # Perpendicular unit normal; choose sign so that + is towards torso (medial) side
            v_len = math.hypot(vx, vy)
            nx_raw, ny_raw = vy / v_len, -vx / v_len  # one of the two perpendiculars
            medial_dir_x = 1.0 if (torso_center[0] - hx) >= 0 else -1.0
            if nx_raw * medial_dir_x < 0:
                nx, ny = -nx_raw, -ny_raw
            else:
                nx, ny = nx_raw, ny_raw
            offset_px = (kx - px) * nx + (ky - py) * ny  # signed offset (+ inward)
            norm_offset = abs(offset_px) / max(v_len, 1e-6)
            inward = offset_px > 0

        return {
            "points": {
                "shoulder_l": shoulder_l,
                "shoulder_r": shoulder_r,
                "hip": hip,
                "knee": knee,
                "ankle": ankle,
                "torso_center": torso_center,
                "torso_ref": torso_ref,
                "knee_proj": (px, py),
            },
            "angles": {
                "knee": knee_angle,
                "hip": hip_angle,
                "torso": torso_angle,
            },
            "valgus": {
                "norm": float(norm_offset),
                "inward": bool(inward),
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

        # Exclude hip from determining overall status to avoid flagging normal hip flexion
        worst = max(level(knee_color), level(torso_color))
        status = "good" if worst == 0 else ("caution" if worst == 1 else "unsafe")
        reasons = []
        if status != "good":
            if level(knee_color) == worst:
                reasons.append("Knee angle out of range")
            if level(torso_color) == worst:
                reasons.append("Torso alignment off")

        # Knee valgus evaluation (only when sufficiently flexed to be meaningful)
        v = metrics.get("valgus", {})  # type: ignore[assignment]
        if isinstance(v, dict):
            norm = float(v.get("norm", 0.0))
            inward = bool(v.get("inward", False))
            if knee < 150.0 and inward:
                if norm >= self.valgus_unsafe_norm:
                    status = "unsafe"
                    reasons.append("Knee valgus (inward collapse)")
                elif norm >= self.valgus_caution_norm and status == "good":
                    status = "caution"
                    reasons.append("Knee valgus (inward collapse)")

        return {
            "status": status,
            "colors": {"knee": knee_color, "hip": hip_color, "torso": torso_color},
            "reasons": reasons,
        }

    def rep_update(self, metrics: Dict[str, object], classification: Optional[Dict[str, object]] = None) -> Dict[str, int]:
        knee_angle = float(metrics["angles"]["knee"])  # type: ignore[index]

        # Determine current status from provided classification or compute it
        status = None
        if classification and isinstance(classification, dict):
            status = classification.get("status")
        if status is None:
            status = self.classify_state(metrics).get("status")

        # While in the down phase of a rep, track if any unsafe occurs
        if self._rep_state == "down":
            if status == "unsafe":
                self._had_unsafe_in_rep = True

        if knee_angle < 100 and self._rep_state == "up":
            # Start of a new rep (descending)
            self._rep_state = "down"
            self._had_unsafe_in_rep = (status == "unsafe")
        elif knee_angle >= 160 and self._rep_state == "down":
            # End of rep at the top; decide correctness based on any unsafe during movement
            if self._had_unsafe_in_rep:
                self._incorrect_reps += 1
            else:
                self._correct_reps += 1
            self._rep_state = "up"
            self._had_unsafe_in_rep = False

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

        # Angle text (hide hip angle per request)
        cv2.putText(frame, f"Knee: {int(angles['knee'])}{self._deg}", (knee[0] - 60, knee[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors["knee"], 2)  # type: ignore[index]
        cv2.putText(frame, f"Torso: {int(angles['torso'])}{self._deg}", (torso_center[0] - 60, torso_center[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors["torso"], 2)  # type: ignore[index]

        # Angle arcs at key joints for clearer visualization
        self._draw_angle_arc(frame, knee, hip, ankle, colors["knee"])  # type: ignore[index]
        self._draw_angle_arc(frame, hip, torso_center, knee, colors["hip"])  # type: ignore[index]
        self._draw_angle_arc(frame, torso_center, torso_ref, hip, colors["torso"])  # type: ignore[index]

        # Knee valgus overlay: show offset from hip-ankle line with caution/unsafe coloring
        valgus = metrics.get("valgus", {})  # type: ignore[assignment]
        if isinstance(valgus, dict):
            norm = float(valgus.get("norm", 0.0))
            inward = bool(valgus.get("inward", False))
            knee_proj = pts.get("knee_proj", None)  # type: ignore[assignment]
            if knee_proj and inward:
                if norm >= self.valgus_unsafe_norm:
                    v_color = (0, 0, 255)
                elif norm >= self.valgus_caution_norm:
                    v_color = (0, 255, 255)
                else:
                    v_color = (0, 255, 0)
                # Reference hip-ankle line
                cv2.line(frame, hip, ankle, (90, 90, 90), 1)  # type: ignore[index]
                # Offset vector from projection to knee
                cv2.line(frame, knee_proj, knee, v_color, 3)  # type: ignore[arg-type]
                # Label near knee
                pct = int(round(norm * 100))
                label = f"Valgus: {pct}%"
                cv2.putText(frame, label, (knee[0] + 10, knee[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, v_color, 2)  # type: ignore[index]

        # HUD: responsive top banner spanning width with left/right aligned text
        banner = classification["status"].upper()  # type: ignore[index]
        banner_color = (0, 200, 0) if banner == "GOOD" else ((0, 200, 200) if banner == "CAUTION" else (0, 0, 200))

        h_frame, w_frame = frame.shape[0], frame.shape[1]
        pad = 8

        # Measure text sizes
        font_large = cv2.FONT_HERSHEY_SIMPLEX
        font_small = cv2.FONT_HERSHEY_SIMPLEX
        banner_scale, banner_th = 1.0, 2
        counts_scale, counts_th = 0.75, 2

        (banner_w, banner_h), _ = cv2.getTextSize(banner, font_large, banner_scale, banner_th)
        rep_text = f"Correct: {counts['correct']}  Incorrect: {counts['incorrect']}"
        (counts_w, counts_h), _ = cv2.getTextSize(rep_text, font_small, counts_scale, counts_th)

        # Draw full-width top bar
        bar_x0, bar_y0 = 14, 14
        bar_x1 = max(bar_x0 + banner_w + 2 * pad, w_frame - 14)
        bar_h = max(banner_h, counts_h) + 2 * pad
        bar_y1 = bar_y0 + bar_h
        cv2.rectangle(frame, (bar_x0, bar_y0), (bar_x1, bar_y1), (30, 30, 30), -1)

        # Left-aligned banner
        banner_x = bar_x0 + pad
        banner_y = bar_y0 + pad + banner_h
        cv2.putText(frame, banner, (banner_x, banner_y), font_large, banner_scale, banner_color, banner_th)

        # Right-aligned counts, clamped inside the bar
        counts_x = max(bar_x0 + pad, min(bar_x1 - pad - counts_w, w_frame - pad - counts_w))
        counts_y = bar_y0 + pad + counts_h
        cv2.putText(frame, rep_text, (counts_x, counts_y), font_small, counts_scale, (255, 255, 255), counts_th)

        # Reasons panel (if any) below the bar, kept within width
        reasons = classification.get("reasons", [])  # type: ignore[assignment]
        if isinstance(reasons, list) and len(reasons) > 0:
            base_y = bar_y1 + 10
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
            # Optional valgus thresholds
            if "valgus_caution_norm" in data:
                self.valgus_caution_norm = float(data["valgus_caution_norm"])  # type: ignore[assignment]
            if "valgus_unsafe_norm" in data:
                self.valgus_unsafe_norm = float(data["valgus_unsafe_norm"])  # type: ignore[assignment]
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


