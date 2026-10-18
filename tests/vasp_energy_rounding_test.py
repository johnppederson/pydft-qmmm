"""Tests for undoing VASP's single-precision plugin energy."""

import numpy as np
import pytest

from pydft_qmmm.interfaces.vasp import vasp_utils


def _lost(x):
    return x - float(np.float32(x))


def test_rounding_uses_force_and_the_local_call_printed_in_outcar(tmp_path):
    """The local-potential value is the one whose float32 VASP printed."""
    # Far enough apart (> one float32 step at 49 eV) to round differently.
    local = [49.44071111, 49.44081234, 49.44091301]
    force = 69.78031234567
    (tmp_path / "PLUGIN_ENERGY").write_text(
        "".join(f"local_potential {x!r}\n" for x in local)
        + f"force_and_stress {force!r}\n",
    )
    # VASP used the middle call for its final energy.
    printed = float(np.float32(local[1]))
    (tmp_path / "OUTCAR").write_text(
        f"  ext. energy  EPLUGINS = {float(np.float32(local[0])):18.8f}\n"
        f"  ext. energy  EPLUGINS = {printed:18.8f}\n",
    )
    expected = _lost(force) + _lost(local[1])
    assert vasp_utils.plugin_energy_rounding(str(tmp_path)) == pytest.approx(expected, abs=1e-12)


def test_rounding_without_local_energy(tmp_path):
    """Only the force callback contributes when no self term is used."""
    force = -123.456789012345
    (tmp_path / "PLUGIN_ENERGY").write_text(
        "local_potential 0.0\nlocal_potential 0.0\n"
        f"force_and_stress {force!r}\n",
    )
    assert vasp_utils.plugin_energy_rounding(str(tmp_path)) == pytest.approx(_lost(force), abs=1e-12)
    assert abs(_lost(force)) > 1e-7


def test_missing_log_raises(tmp_path):
    """A missing log is an error, not a silent zero correction."""
    with pytest.raises(vasp_utils.VaspExecutionError):
        vasp_utils.plugin_energy_rounding(str(tmp_path))


def test_local_calls_after_the_force_callback_are_ignored(tmp_path):
    """VASP's setup call after the force callback is not in the final energy."""
    used, later = 49.44068404447441, 49.4406833879732
    force = 69.78033261605418
    (tmp_path / "PLUGIN_ENERGY").write_text(
        f"local_potential {used!r}\nforce_and_stress {force!r}\nlocal_potential {later!r}\n",
    )
    (tmp_path / "OUTCAR").write_text(
        f"  ext. energy  EPLUGINS = {float(np.float32(used)):18.8f}\n",
    )
    expected = _lost(force) + _lost(used)
    assert vasp_utils.plugin_energy_rounding(str(tmp_path)) == pytest.approx(expected, abs=1e-12)


def test_persistent_frames_use_their_own_entries(tmp_path):
    """Each persistent frame uses its own force entry and its last local call."""
    frames = [(49.4406, 69.7803), (49.5123, 70.1234), (49.3011, 68.9999)]
    lines = []
    for local, force in frames:
        lines += [f"local_potential {local + 1e-3!r}\n", f"local_potential {local!r}\n",
                  f"force_and_stress {force!r}\n", f"local_potential {local - 2e-3!r}\n"]
    (tmp_path / "PLUGIN_ENERGY").write_text("".join(lines))
    for frame, (local, force) in enumerate(frames):
        expected = _lost(force) + _lost(local)
        assert vasp_utils.plugin_energy_rounding(str(tmp_path), frame=frame) == pytest.approx(expected, abs=1e-12)
    with pytest.raises(vasp_utils.VaspExecutionError):
        vasp_utils.plugin_energy_rounding(str(tmp_path), frame=3)
