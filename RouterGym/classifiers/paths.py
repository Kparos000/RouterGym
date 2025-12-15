from __future__ import annotations

from pathlib import Path

# Absolute path to the calibrated encoder head artifact.
HEAD_PATH = Path(__file__).with_name("encoder_calibrated_head.npz")

__all__ = ["HEAD_PATH"]
