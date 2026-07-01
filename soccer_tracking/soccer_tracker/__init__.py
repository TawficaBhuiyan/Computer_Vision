"""
soccer_tracker
==============
A modular BoT-SORT + C-BIoU multi-object tracker for football footage.

Public API:
    from soccer_tracker import load_config, BoTSORTCBIoUTracker, VideoTrackingPipeline

Note: the core tracker (config + BoTSORTCBIoUTracker) depends only on
numpy / scipy / opencv. `VideoTrackingPipeline` additionally needs
ultralytics + supervision, so it is imported lazily - you can use and test the
tracker without the heavy detection stack installed.
"""

from .config import load_config
from .tracker import BoTSORTCBIoUTracker

__all__ = ["load_config", "BoTSORTCBIoUTracker", "VideoTrackingPipeline"]
__version__ = "1.0.0"


def __getattr__(name):
    # Lazy import so importing the package (or running tests) does not require
    # ultralytics/supervision unless the full video pipeline is actually used.
    if name == "VideoTrackingPipeline":
        from .pipeline import VideoTrackingPipeline
        return VideoTrackingPipeline
    raise AttributeError(f"module 'soccer_tracker' has no attribute '{name}'")
