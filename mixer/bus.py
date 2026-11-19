"""Core mixing-bus primitives for the audio-mixer challenge.

Audio is always represented as ``list[list[float]]``: the outer list holds
channels and each inner list holds mono frames in playback order
(channel-major).  Every sample is a float in ``[-1.0, 1.0]`` where 0 dBFS is
1.0.  Frame counts and sample rates are explicit caller-side facts; this module
never reads a clock, never touches the network, never sleeps and never uses
third-party packages.

Contract notes
--------------
* Amplitude gain and decibels are related by ``gain = 10 ** (db / 20)``
  (20*log10, not the 10*log10 power relation), so -6 dB is 0.501187... and
  +6 dB is 1.99526....
* ``gain_to_db(0.0)`` is ``-inf`` (silence); negative or non-finite gains raise
  :class:`MixError`.
* Float assertions in the test suite use ``pytest.approx(..., abs=1e-6)``
  unless a tighter round-trip tolerance (1e-9) is stated.
* Invalid arguments (out-of-range pan, unknown pan law, negative latency,
  unknown layout, non-finite audio, non-finite master gain) raise
  :class:`MixError`; no function silently drops a channel.
"""

from __future__ import annotations

import math
import random


# Channel counts for the layouts supported by :func:`convert_layout`.
_LAYOUT_CHANNELS = {"mono": 1, "stereo": 2, "5.1": 6}


class MixError(ValueError):
    """Raised when a mixer argument or audio job violates its contract."""


def _is_real_number(value: object) -> bool:
    """Return True for finite int/float samples (booleans excluded)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _validate_audio(samples: list[list[float]]) -> None:
    """Raise :class:`MixError` for malformed structures or non-finite samples."""
    if not isinstance(samples, list):
        raise MixError("audio must be a list of channels")
    for channel in samples:
        if not isinstance(channel, list):
            raise MixError("each channel must be a list of frames")
        for sample in channel:
            if not _is_real_number(sample):
                raise MixError("all samples must be finite real numbers")


def db_to_gain(db: float) -> float:
    """Convert a level in decibels to a linear amplitude gain.

    Uses the amplitude relation ``gain = 10 ** (db / 20)`` (20*log10, not the
    10*log10 power relation used for energy).  ``db_to_gain(0.0)`` is exactly
    ``1.0`` and ``db_to_gain(-6.0)`` is ``0.5011872336...``; samples are
    multiplied by the returned linear value.

    Args:
        db: Level in decibels.  Must be finite; fractional values are allowed.

    Returns:
        Linear gain ``10 ** (db / 20.0)``, always positive and finite for a
        finite ``db``.

    Raises:
        MixError: If ``db`` is NaN or infinite.
    """
    if not _is_real_number(db):
        raise MixError("db must be a finite real number")
    return 10.0 ** (db / 20.0)


def gain_to_db(gain: float) -> float:
    """Convert a linear amplitude gain to decibels with ``20 * log10(gain)``.

    ``gain_to_db(0.0)`` returns ``-inf`` (silence).  Positive gains round-trip
    through :func:`db_to_gain`; the test suite requires an absolute error of at
    most 1e-9 for that round trip.

    Args:
        gain: Linear amplitude gain.  ``0.0`` is allowed; negative and
            non-finite gains are not.

    Returns:
        ``20.0 * math.log10(gain)``, or ``-inf`` when ``gain == 0.0``.

    Raises:
        MixError: If ``gain`` is negative or non-finite.
    """
    if not isinstance(gain, (int, float)) or isinstance(gain, bool) or not math.isfinite(gain):
        raise MixError("gain must be a finite real number")
    if gain < 0.0:
        raise MixError("gain must not be negative")
    if gain == 0.0:
        return -math.inf
    return 20.0 * math.log10(gain)


def pan_to_gains(pan: float, law: str = "linear") -> tuple[float, float]:
    """Map a stereo pan position to ``(left_gain, right_gain)``.

    ``pan`` is validated, not clamped: it must already be in ``[-1.0, 1.0]``
    (-1 is hard left, 0 is centre, +1 is hard right).

    Supported laws:

    ``"linear"``
        ``left = (1 - pan) / 2`` and ``right = (1 + pan) / 2``.  Centre is
        ``(0.5, 0.5)`` and the two gains sum to 1.
    ``"-3db"``
        Equal-power (constant-power) pan with ``theta = (pan + 1) * pi / 4``,
        ``left = cos(theta)`` and ``right = sin(theta)``.  Centre is
        ``(sqrt(0.5), sqrt(0.5))``, i.e. each side is -3 dB relative to a
        hard-panned channel, and ``left**2 + right**2 == 1``.

    Returns:
        A ``tuple`` of two non-negative linear gains.

    Raises:
        MixError: If ``pan`` is outside ``[-1, 1]`` or non-finite, or if
            ``law`` is not one of the supported strings.
    """
    if not _is_real_number(pan) or not (-1.0 <= pan <= 1.0):
        raise MixError("pan must be a finite real number in [-1.0, 1.0]")
    if law == "linear":
        return (1.0 - pan) / 2.0, (1.0 + pan) / 2.0
    if law == "-3db":
        theta = (pan + 1.0) * math.pi / 4.0
        return math.cos(theta), math.sin(theta)
    raise MixError(f"unknown pan law: {law!r}")


def mix_buses(
    buses: list[list[list[float]]], master_gain: float = 1.0
) -> list[list[float]]:
    """Sum several buses into one bus using channel-major layout.

    Each bus is ``list[list[float]]`` (channels x frames).  All buses belong to
    the same layout, so they are aligned at frame 0 and summed sample by
    sample.  The output follows the widest bus in the batch:

    * channel count is the maximum number of channels across ``buses``; buses
      with fewer channels are zero-padded (silent);
    * frame count is the maximum frame count across all channels; shorter
      channels are zero-padded at the tail, so a long bus never loses its tail.

    Summation here is purely linear, with no limiter: chain
    :func:`apply_limiter` afterwards when the result must stay inside full
    scale.  ``master_gain`` multiplies every summed sample.

    Args:
        buses: Bus list; an empty list is legal.
        master_gain: Linear gain applied once to the summed bus.  Must be
            finite.

    Returns:
        ``list[list[float]]`` with the summed channels, or ``[]`` when
        ``buses`` is empty (the only silence expressible without a layout).

    Raises:
        MixError: If ``master_gain`` or any sample is non-finite, or if the
            bus structure is malformed.
    """
    if not _is_real_number(master_gain):
        raise MixError("master_gain must be a finite real number")
    if not isinstance(buses, list):
        raise MixError("buses must be a list")
    if not buses:
        return []

    # Validate the whole batch before publishing any summed sample.
    for bus in buses:
        _validate_audio(bus)

    channel_count = max(len(bus) for bus in buses)
    frame_count = 0
    for bus in buses:
        for channel in bus:
            if len(channel) > frame_count:
                frame_count = len(channel)

    mixed = [[0.0] * frame_count for _ in range(channel_count)]
    for bus in buses:
        for channel_index, channel in enumerate(bus):
            target = mixed[channel_index]
            for frame_index, sample in enumerate(channel):
                target[frame_index] += sample

    if master_gain != 1.0:
        mixed = [
            [sample * master_gain for sample in channel] for channel in mixed
        ]
    return mixed


def apply_limiter(
    samples: list[list[float]], ceiling: float = 1.0
) -> list[list[float]]:
    """Soft-limit ``samples`` so every output magnitude stays within ``ceiling``.

    The limiter is linear below ``ceiling / 2`` and applies a smooth, monotone
    knee above it, so a summed peak such as 1.6 at ``ceiling=1.0`` comes out
    just below 1.0 instead of hard-clipping.  Two hard guarantees: no output
    magnitude exceeds ``ceiling`` and values already inside the linear region
    are returned unchanged (within 1e-6).

    ``ceiling == 0.0`` is legal and returns an all-zero (silent) bus with the
    same channel/frame shape.  Negative or non-finite ceilings and non-finite
    samples raise :class:`MixError`.

    Args:
        samples: ``list[list[float]]`` channels x frames.
        ceiling: Non-negative full-scale bound.

    Returns:
        A new ``list[list[float]]`` with the same channel/frame shape.

    Raises:
        MixError: If ``ceiling`` is negative/non-finite or a sample is
            non-finite.
    """
    if not isinstance(ceiling, (int, float)) or isinstance(ceiling, bool) or not math.isfinite(ceiling):
        raise MixError("ceiling must be a finite real number")
    if ceiling < 0.0:
        raise MixError("ceiling must not be negative")
    _validate_audio(samples)

    if ceiling == 0.0:
        return [[0.0] * len(channel) for channel in samples]

    threshold = ceiling / 2.0

    def knee(value: float) -> float:
        magnitude = abs(value)
        if magnitude <= threshold:
            return value
        # Rational soft knee: joins the unity-gain line at the threshold with
        # matching value and slope and asymptotically approaches ``ceiling``.
        limited = ceiling - (ceiling * ceiling) / (4.0 * magnitude)
        return limited if value > 0.0 else -limited

    return [[knee(sample) for sample in channel] for channel in samples]


def compensate_latency(
    channels: list[list[float]], delay_frames: int = 0
) -> list[list[float]]:
    """Remove a known processing latency of ``delay_frames`` frames.

    The signal is shifted earlier by ``delay_frames`` frames: the first
    ``delay_frames`` frames are discarded, the remaining frames move to the
    start and the tail is zero-padded, so every channel keeps its original
    frame count.  ``delay_frames == 0`` is an exact identity, and a delay
    longer than a channel yields all zeros for that channel.

    Frame counts are explicit integers; nothing here consults a sample rate or
    sleeps.

    Args:
        channels: ``list[list[float]]`` channels x frames.
        delay_frames: Non-negative integer latency to compensate.

    Returns:
        A new ``list[list[float]]`` with the same shape, aligned to frame 0.

    Raises:
        MixError: If ``delay_frames`` is negative or not an integer.
    """
    if not isinstance(delay_frames, int) or isinstance(delay_frames, bool):
        raise MixError("delay_frames must be an integer")
    if delay_frames < 0:
        raise MixError("delay_frames must not be negative")
    _validate_audio(channels)

    if delay_frames == 0:
        return [list(channel) for channel in channels]

    aligned = []
    for channel in channels:
        shifted = channel[delay_frames:]
        pad_count = len(channel) - len(shifted)
        aligned.append(list(shifted) + [0.0] * pad_count)
    return aligned


def dither(
    samples: list[list[float]], bit_depth: int = 16, seed: int = 0
) -> list[list[float]]:
    """Quantize to ``bit_depth`` with deterministic TPDF dither.

    A fresh ``random.Random(seed)`` instance drives triangular-PDF noise of
    +/-1 LSB (the sum of two uniform draws).  The noise is added before
    rounding to the quantizer step ``2 / 2**bit_depth`` and the result is
    clamped to ``[-1.0, 1.0]``.  Therefore:

    * equal ``seed`` plus equal input produce bit-identical output (no global
      ``random`` state, no wall clock);
    * an out-of-range input such as -1.2 clamps to the negative full-scale
      rail and never wraps around into positive noise;
    * every returned sample lies inside ``[-1.0, 1.0]``.

    Args:
        samples: ``list[list[float]]`` channels x frames.
        bit_depth: Integer in ``[1, 32]`` (16 = CD, 24 = studio master).
        seed: Integer seed for the TPDF generator.

    Returns:
        A new quantized ``list[list[float]]`` with the same shape.

    Raises:
        MixError: If ``bit_depth`` is out of range, ``seed`` is not an integer
            or any sample is non-finite.
    """
    if not isinstance(bit_depth, int) or isinstance(bit_depth, bool) or not (1 <= bit_depth <= 32):
        raise MixError("bit_depth must be an integer in [1, 32]")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise MixError("seed must be an integer")
    _validate_audio(samples)

    rng = random.Random(seed)
    lsb = 2.0 / (2 ** bit_depth)
    quantized = []
    for channel in samples:
        output_channel = []
        for sample in channel:
            # TPDF: two uniform [0, 1) draws centred to [-1, 1) LSB.
            noise = (rng.random() + rng.random() - 1.0) * lsb
            steps = math.floor((sample + noise) / lsb + 0.5)
            value = steps * lsb
            if value < -1.0:
                value = -1.0
            elif value > 1.0:
                value = 1.0
            output_channel.append(value)
        quantized.append(output_channel)
    return quantized


def convert_layout(
    samples: list[list[float]], from_layout: str, to_layout: str
) -> list[list[float]]:
    """Convert channel-major audio between ``mono``, ``stereo`` and ``5.1``.

    Layout order is fixed by this contract:

    * ``mono``   -> ``[M]``
    * ``stereo`` -> ``[L, R]``
    * ``5.1``    -> ``[L, R, C, LFE, Ls, Rs]``

    Matrix rules:

    * mono -> stereo: ``L = R = M``.
    * stereo -> mono: ``M = (L + R) / 2``.
    * mono -> 5.1: ``C = M``; every other channel stays silent, LFE included.
    * stereo -> 5.1: ``L`` and ``R`` are copied; ``C = (L + R) / 2``; LFE and
      both surrounds stay silent.
    * 5.1 -> stereo: ``L' = L + sqrt(0.5) * C + sqrt(0.5) * Ls`` and
      ``R' = R + sqrt(0.5) * C + sqrt(0.5) * Rs``; LFE is discarded and never
      summed into L/R.
    * 5.1 -> mono: the same fold-down as stereo, then ``(L' + R') / 2``.
    * identical layout names: an exact copy with no gain change.

    The LFE rule is deliberate: LFE carries low-frequency effects and is not a
    regular channel, so it must not add bass (and roughly +6 dB when combined
    with centre) to a stereo fold-down.  Conversion is pure gain routing;
    limiting is a separate stage.

    Args:
        samples: ``list[list[float]]`` channels x frames.
        from_layout: One of ``"mono"``, ``"stereo"``, ``"5.1"``.
        to_layout: One of ``"mono"``, ``"stereo"``, ``"5.1"``.

    Returns:
        A new ``list[list[float]]`` in ``to_layout`` channel order.

    Raises:
        MixError: If a layout name is unknown or ``samples`` has a channel
            count that does not match ``from_layout``.
    """
    if from_layout not in _LAYOUT_CHANNELS:
        raise MixError(f"unknown source layout: {from_layout!r}")
    if to_layout not in _LAYOUT_CHANNELS:
        raise MixError(f"unknown target layout: {to_layout!r}")
    _validate_audio(samples)
    if len(samples) != _LAYOUT_CHANNELS[from_layout]:
        raise MixError(
            f"{from_layout!r} audio must have {_LAYOUT_CHANNELS[from_layout]} "
            f"channels, got {len(samples)}"
        )

    frame_count = len(samples[0]) if samples else 0
    zeros = [0.0] * frame_count

    if from_layout == to_layout:
        return [list(channel) for channel in samples]

    if from_layout == "mono":
        mono = samples[0]
        if to_layout == "stereo":
            return [list(mono), list(mono)]
        # mono -> 5.1: M routes to the centre channel, never to LFE.
        return [list(zeros), list(zeros), list(mono), list(zeros), list(zeros), list(zeros)]

    if from_layout == "stereo":
        left, right = samples
        if to_layout == "mono":
            return [[(l + r) / 2.0 for l, r in zip(left, right)]]
        # stereo -> 5.1
        centre = [(l + r) / 2.0 for l, r in zip(left, right)]
        return [
            list(left),
            list(right),
            centre,
            list(zeros),
            list(zeros),
            list(zeros),
        ]

    # from_layout == "5.1": semantic indices [L, R, C, LFE, Ls, Rs].
    left, right, centre, _lfe, surround_left, surround_right = samples
    attenuated = math.sqrt(0.5)
    fold_left = [
        l + attenuated * c + attenuated * ls
        for l, c, ls in zip(left, centre, surround_left)
    ]
    fold_right = [
        r + attenuated * c + attenuated * rs
        for r, c, rs in zip(right, centre, surround_right)
    ]
    if to_layout == "stereo":
        return [fold_left, fold_right]
    # 5.1 -> mono
    return [[(l + r) / 2.0 for l, r in zip(fold_left, fold_right)]]


def mix_batch(
    jobs: list[dict], master_gain: float = 1.0
) -> dict[str, list[list[float]]]:
    """Mix a batch of jobs with all-or-nothing validation.

    Each job is a mapping with:

    * ``"id"``: unique string key for the returned mapping;
    * ``"buses"``: a bus list accepted by :func:`mix_buses`.

    Every job is mixed with the same ``master_gain``.  Validation happens for
    the whole batch before any result is published: if one job contains NaN or
    +/-inf (in a sample, in a bus structure, or in ``master_gain``), the call
    raises :class:`MixError` and returns nothing instead of leaving a partially
    mixed batch or a poisoned NaN entry downstream (whole-batch rollback).

    On success the returned ``dict`` preserves input job order (Python 3.7+
    insertion order), so callers can re-zip it with ``jobs``.

    Args:
        jobs: List of job mappings, each with an ``"id"`` and a ``"buses"``
            list.
        master_gain: Linear gain applied to every job.  Must be finite.

    Returns:
        ``{job_id: mixed_bus}`` with keys in the same order as ``jobs``.

    Raises:
        MixError: If ``master_gain`` is non-finite, any sample is non-finite,
            a job is malformed, or ids are duplicated.
    """
    if not _is_real_number(master_gain):
        raise MixError("master_gain must be a finite real number")
    if not isinstance(jobs, list):
        raise MixError("jobs must be a list")

    seen_ids: set[str] = set()
    for job in jobs:
        if not isinstance(job, dict):
            raise MixError("each job must be a mapping")
        job_id = job.get("id")
        buses = job.get("buses")
        if not isinstance(job_id, str):
            raise MixError("each job must have a string 'id'")
        if not isinstance(buses, list):
            raise MixError(f"job {job_id!r} must have a list of buses")
        if job_id in seen_ids:
            raise MixError(f"duplicate job id: {job_id!r}")
        seen_ids.add(job_id)

    # Mix every job first; a raised MixError discards the whole local result,
    # so no partial batch or NaN-bearing entry is ever published.
    results: dict[str, list[list[float]]] = {}
    for job in jobs:
        results[job["id"]] = mix_buses(job["buses"], master_gain)
    return results
