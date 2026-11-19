"""Public surface of the audio mixing-bus package.

Import the primitives from here (``from mixer import mix_buses``) or directly
from :mod:`mixer.bus`.  The implementation lives in ``mixer/bus.py``.
"""

from __future__ import annotations

from .bus import (
    MixError,
    apply_limiter,
    compensate_latency,
    convert_layout,
    db_to_gain,
    dither,
    gain_to_db,
    mix_batch,
    mix_buses,
    pan_to_gains,
)

__all__ = [
    "MixError",
    "apply_limiter",
    "compensate_latency",
    "convert_layout",
    "db_to_gain",
    "dither",
    "gain_to_db",
    "mix_batch",
    "mix_buses",
    "pan_to_gains",
]
