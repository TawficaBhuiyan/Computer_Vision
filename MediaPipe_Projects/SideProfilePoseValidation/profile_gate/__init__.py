"""Real-Time Side Profile Full Body Pose Validation System."""
from .config import ProfileGateConfig, Orientation
from .detector import PoseDetector
from .gate import ProfileGate, GateResult
from .render import draw

__all__ = [
    "ProfileGateConfig", "Orientation", "PoseDetector",
    "ProfileGate", "GateResult", "draw",
]
__version__ = "1.0.0"
