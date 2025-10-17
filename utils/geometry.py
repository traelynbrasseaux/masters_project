import math
from typing import Dict, Tuple, Iterable, Union


def calculate_angle(a: Tuple[int, int], b: Tuple[int, int], c: Tuple[int, int]) -> float:
    """Calculate the angle at point b formed by points a-b-c in degrees.

    The function is robust to near-collinear vectors via clamping.
    """
    ax, ay = a
    bx, by = b
    cx, cy = c

    ba_x, ba_y = ax - bx, ay - by
    bc_x, bc_y = cx - bx, cy - by

    ba_norm = math.hypot(ba_x, ba_y)
    bc_norm = math.hypot(bc_x, bc_y)
    if ba_norm == 0.0 or bc_norm == 0.0:
        return 0.0

    cosine = (ba_x * bc_x + ba_y * bc_y) / (ba_norm * bc_norm)
    cosine = max(min(cosine, 1.0), -1.0)
    return math.degrees(math.acos(cosine))


def in_ranges(value: float, ranges: Union[Tuple[float, float], Iterable[Tuple[float, float]]]) -> bool:
    """Return True if value is within the provided range or any of the ranges.

    Accepts either a single (lo, hi) tuple or an iterable of such tuples.
    """
    if not hasattr(ranges, "__iter__"):
        return False
    # Detect single range
    if isinstance(ranges, tuple) and len(ranges) == 2 and not isinstance(ranges[0], tuple):
        lo, hi = ranges
        return lo <= value <= hi
    # Multiple ranges
    return any(lo <= value <= hi for (lo, hi) in ranges)  # type: ignore[arg-type]


def get_angle_color(angle: float, safe_range: Tuple[float, float], caution_ranges: Union[Tuple[float, float], Iterable[Tuple[float, float]]]) -> Tuple[int, int, int]:
    """Map an angle to a BGR color by safety zone: green=Safe, yellow=Caution, red=Unsafe."""
    if safe_range[0] <= angle <= safe_range[1]:
        return (0, 255, 0)
    if in_ranges(angle, caution_ranges):
        return (0, 255, 255)
    return (0, 0, 255)


class EMA:
    """Simple exponential moving average smoother keyed by metric name."""

    def __init__(self, alpha: float = 0.4) -> None:
        self.alpha: float = alpha
        self._state: Dict[str, float] = {}

    def reset(self) -> None:
        self._state.clear()

    def apply(self, key: str, value: float) -> float:
        prev = self._state.get(key)
        if prev is None:
            self._state[key] = value
            return value
        smoothed = self.alpha * value + (1.0 - self.alpha) * prev
        self._state[key] = smoothed
        return smoothed


