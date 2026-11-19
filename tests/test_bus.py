"""Contract tests for ``mixer.bus`` -- 15 tests, all initially red.

Audio convention: ``list[list[float]]`` is channel-major (channels x frames),
each sample is a float in ``[-1.0, 1.0]`` (0 dBFS == 1.0).  Frame counts are
explicit; there is no random sampling, no sleeping, no network and no external
dependency.

The implementation in ``mixer/bus.py`` is a set of ``NotImplementedError``
stubs, so the expected first run is exactly ``15 failed`` with no collection,
import or syntax errors.  This file is the acceptance spec and must not be
modified while fixing the mixer.
"""

from __future__ import annotations

import math

import pytest

from mixer.bus import (
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


# ---------------------------------------------------------------------------
# One test per documented defect (10)
# ---------------------------------------------------------------------------


def test_db_conversion_uses_twenty_log10():
    """Defect 1: 10*log10 would turn -6 dB into -3 dB; use 20*log10."""
    assert db_to_gain(-6.0) == pytest.approx(0.5011872336272722, abs=1e-6)
    assert db_to_gain(6.0) == pytest.approx(1.9952623149688795, abs=1e-6)
    assert gain_to_db(0.5) == pytest.approx(-6.020599913279624, abs=1e-6)
    with pytest.raises(MixError):
        gain_to_db(-0.1)


def test_pan_equal_power_law_is_minus_three_db_at_centre():
    """Defect 2: the -3db law must be equal-power, not linear."""
    left, right = pan_to_gains(0.0, "-3db")
    assert left == pytest.approx(math.sqrt(0.5), abs=1e-6)
    assert right == pytest.approx(math.sqrt(0.5), abs=1e-6)
    assert left * left + right * right == pytest.approx(1.0, abs=1e-6)

    lin_left, lin_right = pan_to_gains(0.0, "linear")
    assert lin_left == pytest.approx(0.5, abs=1e-6)
    assert lin_right == pytest.approx(0.5, abs=1e-6)

    hard_left, hard_right = pan_to_gains(-1.0, "-3db")
    assert hard_left == pytest.approx(1.0, abs=1e-6)
    assert hard_right == pytest.approx(0.0, abs=1e-6)

    with pytest.raises(MixError):
        pan_to_gains(1.5, "linear")
    with pytest.raises(MixError):
        pan_to_gains(0.0, "quadraphonic")


def test_mix_buses_keeps_the_longest_tail():
    """Defect 3: mixing must not truncate the batch to its shortest channel."""
    short_bus = [[1.0, 1.0, 1.0, 1.0]]
    long_bus = [[0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]

    mixed = mix_buses([short_bus, long_bus], 1.0)

    assert len(mixed) == 1
    assert len(mixed[0]) == 6
    assert mixed[0][:4] == pytest.approx([1.5, 1.5, 1.5, 1.5], abs=1e-6)
    assert mixed[0][4:] == pytest.approx([0.5, 0.5], abs=1e-6)


def test_summed_bus_is_soft_limited_within_full_scale():
    """Defect 4: a summed peak above 1.0 must be limited, not played clipped."""
    hot = [[0.8, -0.8, 0.2, -0.2]]
    summed = mix_buses([hot, hot], 1.0)
    assert summed[0][0] == pytest.approx(1.6, abs=1e-6)

    limited = apply_limiter(summed, 1.0)

    assert len(limited) == 1
    assert len(limited[0]) == 4
    assert all(-1.0 <= value <= 1.0 for value in limited[0])


def test_compensate_latency_aligns_frames():
    """Defect 5: latency compensation must shift frames, not pass them through."""
    aligned = compensate_latency([[10.0, 20.0, 30.0, 40.0, 50.0]], 2)
    assert len(aligned) == 1
    assert aligned[0] == pytest.approx([30.0, 40.0, 50.0, 0.0, 0.0], abs=1e-6)

    assert compensate_latency([[1.0, 2.0]], 0)[0] == pytest.approx(
        [1.0, 2.0], abs=1e-6
    )
    assert compensate_latency([[1.0, 2.0]], 5)[0] == pytest.approx(
        [0.0, 0.0], abs=1e-6
    )
    with pytest.raises(MixError):
        compensate_latency([[1.0, 2.0]], -1)


def test_dither_is_deterministic_for_a_fixed_seed():
    """Defect 6: dither must use a local seeded PRNG, not global random."""
    samples = [[0.0, 0.1, -0.1, 0.49]]
    first = dither(samples, 16, 12345)
    second = dither(samples, 16, 12345)
    other = dither(samples, 16, 54321)

    assert first == second
    assert first != other
    assert all(-1.0 <= value <= 1.0 for channel in first for value in channel)


def test_51_to_stereo_drops_lfe_and_attenuates_centre():
    """Defect 7: 5.1 downmix must not add LFE and must use -3 dB centre."""
    centre_only = [[0.0], [0.0], [1.0], [0.0], [0.0], [0.0]]
    stereo = convert_layout(centre_only, "5.1", "stereo")
    assert len(stereo) == 2
    assert stereo[0] == pytest.approx([math.sqrt(0.5)], abs=1e-6)
    assert stereo[1] == pytest.approx([math.sqrt(0.5)], abs=1e-6)

    lfe_only = [[0.0], [0.0], [0.0], [1.0], [0.0], [0.0]]
    stereo_lfe = convert_layout(lfe_only, "5.1", "stereo")
    assert stereo_lfe[0] == pytest.approx([0.0], abs=1e-6)
    assert stereo_lfe[1] == pytest.approx([0.0], abs=1e-6)

    centre_and_lfe = [[0.0], [0.0], [1.0], [1.0], [0.0], [0.0]]
    stereo_both = convert_layout(centre_and_lfe, "5.1", "stereo")
    assert stereo_both[0] == pytest.approx([math.sqrt(0.5)], abs=1e-6)
    assert stereo_both[1] == pytest.approx([math.sqrt(0.5)], abs=1e-6)


def test_quantization_clamps_instead_of_wrapping():
    """Defect 8: -1.2 must clamp to the negative rail, not wrap positive."""
    quantized = dither([[-1.2, 1.2, 0.0]], 8, 2024)

    assert len(quantized) == 1
    assert len(quantized[0]) == 3
    assert all(-1.0 <= value <= 1.0 for value in quantized[0])
    assert quantized[0][0] < -0.9
    assert quantized[0][1] > 0.9


def test_mix_batch_rolls_back_when_any_sample_is_non_finite():
    """Defect 9: one NaN sample must fail the whole batch, not leak NaN out."""
    jobs = [
        {"id": "clean", "buses": [[[0.1, 0.2]]]},
        {"id": "poisoned", "buses": [[[0.3, float("nan")]]]},
    ]

    with pytest.raises(MixError):
        mix_batch(jobs, 1.0)


def test_mono_to_51_routes_to_centre_not_lfe():
    """Defect 10: mono expansion must put M in C, never in the LFE channel."""
    converted = convert_layout([[0.5, -0.5]], "mono", "5.1")

    assert len(converted) == 6
    assert converted[0] == pytest.approx([0.0, 0.0], abs=1e-6)  # L
    assert converted[1] == pytest.approx([0.0, 0.0], abs=1e-6)  # R
    assert converted[2] == pytest.approx([0.5, -0.5], abs=1e-6)  # C
    assert converted[3] == pytest.approx([0.0, 0.0], abs=1e-6)  # LFE
    assert converted[4] == pytest.approx([0.0, 0.0], abs=1e-6)  # Ls
    assert converted[5] == pytest.approx([0.0, 0.0], abs=1e-6)  # Rs


# ---------------------------------------------------------------------------
# Additional invariants listed by the task (5)
# ---------------------------------------------------------------------------


def test_zero_db_is_unity_gain():
    assert db_to_gain(0.0) == pytest.approx(1.0, abs=1e-12)


def test_db_gain_round_trip():
    assert gain_to_db(0.0) == -math.inf

    for db in (-60.0, -6.0, 0.0, 6.0, 12.0):
        assert gain_to_db(db_to_gain(db)) == pytest.approx(db, abs=1e-9)

    for gain in (0.001, 0.5, 1.0, 2.0, 10.0):
        assert db_to_gain(gain_to_db(gain)) == pytest.approx(gain, rel=1e-9)


def test_empty_bus_returns_silence():
    assert mix_buses([], 1.0) == []
    assert mix_buses([], 0.25) == []


def test_zero_ceiling_limiter_returns_silence():
    limited = apply_limiter([[0.7, -0.7], [1.0, -1.0]], 0.0)
    assert limited == [[0.0, 0.0], [0.0, 0.0]]

    with pytest.raises(MixError):
        apply_limiter([[0.1]], -0.5)


def test_mix_batch_preserves_input_order():
    jobs = [
        {"id": "vocals", "buses": [[[0.5, 0.25]]]},
        {"id": "drums", "buses": [[[0.1, 0.05]]]},
        {"id": "bass", "buses": [[[-0.2, -0.1]]]},
    ]

    mixed = mix_batch(jobs, 1.0)

    assert list(mixed) == ["vocals", "drums", "bass"]
    assert mixed["vocals"][0] == pytest.approx([0.5, 0.25], abs=1e-6)
    assert mixed["drums"][0] == pytest.approx([0.1, 0.05], abs=1e-6)
    assert mixed["bass"][0] == pytest.approx([-0.2, -0.1], abs=1e-6)
