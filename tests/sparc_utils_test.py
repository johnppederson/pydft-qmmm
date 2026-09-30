from __future__ import annotations

import numpy as np
import pytest

from pydft_qmmm.interfaces.sparc import sparc_utils


def test_grid_file_round_trip(tmp_path):
    rng = np.random.default_rng(11)
    field = rng.normal(size=(5, 4, 3))
    cell = np.diag([12.0, 10.0, 8.0])
    path = str(tmp_path / "grid.bin")
    sparc_utils.write_grid_file(path, field, cell, 7)
    got, got_cell, step = sparc_utils.read_grid_file(path)
    assert got == pytest.approx(field)
    assert got_cell == pytest.approx(cell)
    assert step == 7


def test_grid_file_payload_is_fortran_ordered(tmp_path):
    # SPARC indexes as i + j*Nx + k*Nx*Ny, so element [1,0,0] must be
    # the SECOND value in the payload, not the (Ny*Nz)-th.
    field = np.zeros((3, 2, 2))
    field[1, 0, 0] = 5.0
    path = str(tmp_path / "grid.bin")
    sparc_utils.write_grid_file(path, field, np.eye(3), 0)
    with open(path, "rb") as fh:
        fh.readline()
        fh.readline()
        payload = np.frombuffer(fh.read(), dtype="<f8")
    assert payload[1] == pytest.approx(5.0)


def test_read_grid_file_rejects_wrong_shape(tmp_path):
    path = str(tmp_path / "grid.bin")
    sparc_utils.write_grid_file(path, np.zeros((3, 2, 2)), np.eye(3), 0)
    with pytest.raises(ValueError, match="grid"):
        sparc_utils.read_grid_file(path, expect_shape=(4, 2, 2))


def test_read_grid_file_rejects_stale_step(tmp_path):
    path = str(tmp_path / "grid.bin")
    sparc_utils.write_grid_file(path, np.zeros((2, 2, 2)), np.eye(3), 3)
    with pytest.raises(ValueError, match="step"):
        sparc_utils.read_grid_file(path, expect_step=4)


def test_assert_embedding_capable_rejects_stock_binary(tmp_path):
    out = tmp_path / "SPARC.out"
    out.write_text("SPARC output\nPRINT_DENSITY: 0\nFree energy: -1.0\n")
    with pytest.raises(sparc_utils.SparcExecutionError, match="QMMM_FLAG"):
        sparc_utils.assert_embedding_capable(str(out), str(tmp_path))


def test_assert_embedding_capable_accepts_forked_binary(tmp_path):
    out = tmp_path / "SPARC.out"
    out.write_text("SPARC output\nQMMM_FLAG: 1\nFree energy: -1.0\n")
    sparc_utils.assert_embedding_capable(str(out), str(tmp_path))


def test_assert_scf_converged_detects_failure(tmp_path):
    out = tmp_path / "SPARC.out"
    out.write_text("did not converge to desired accuracy\n")
    with pytest.raises(sparc_utils.SparcExecutionError, match="converge"):
        sparc_utils.assert_scf_converged(str(out), str(tmp_path))


# A finished run's SPARC.out has this overall shape: a header, a block
# of SCF iteration lines, an energy/force summary, and a Timing info
# block whose last line is "Total walltime". Modeled on the committed
# example at examples/case_3_sparc/sparc_converge/sparc_workdir/SPARC.out.
_FINISHED_SPARC_OUT = """\
***************************************************************************
*                       SPARC (version May 11, 2023)                      *
***************************************************************************
FD_GRID: 60 60 60
FD_ORDER: 12
BC: P P P
45           -5.9164142111E+00        3.074E-06        2.507
46           -5.9164142120E+00        2.012E-06        2.462
47           -5.9164142206E+00        9.599E-07        2.462
Total number of SCF: 47
====================================================================
                    Energy and force calculation
====================================================================
Free energy per atom               : -5.9164142206E+00 (Ha/atom)
Total free energy                  : -1.7749242662E+01 (Ha)
RMS force                          :  7.1349842152E-03 (Ha/Bohr)
Maximum force                      :  9.5522775837E-03 (Ha/Bohr)
***************************************************************************
                               Timing info
***************************************************************************
Total walltime                     :  131.988 sec
___________________________________________________________________________
"""

# The same run, but with the process killed right after the last SCF
# iteration and before the energy/force summary or Timing info block
# were ever written -- no failure message, no completion marker.
_TRUNCATED_SPARC_OUT = """\
***************************************************************************
*                       SPARC (version May 11, 2023)                      *
***************************************************************************
FD_GRID: 60 60 60
FD_ORDER: 12
BC: P P P
45           -5.9164142111E+00        3.074E-06        2.507
46           -5.9164142120E+00        2.012E-06        2.462
47           -5.9164142206E+00        9.599E-07        2.462
"""


def test_assert_scf_converged_accepts_finished_run(tmp_path):
    out = tmp_path / "SPARC.out"
    out.write_text(_FINISHED_SPARC_OUT)
    sparc_utils.assert_scf_converged(str(out), str(tmp_path))


def test_assert_scf_converged_rejects_empty_file(tmp_path):
    out = tmp_path / "SPARC.out"
    out.write_text("")
    with pytest.raises(sparc_utils.SparcExecutionError, match="walltime"):
        sparc_utils.assert_scf_converged(str(out), str(tmp_path))


def test_assert_scf_converged_rejects_truncated_run(tmp_path):
    out = tmp_path / "SPARC.out"
    out.write_text(_TRUNCATED_SPARC_OUT)
    with pytest.raises(sparc_utils.SparcExecutionError, match="walltime"):
        sparc_utils.assert_scf_converged(str(out), str(tmp_path))


def test_assert_scf_converged_rejects_missing_file(tmp_path):
    out = tmp_path / "SPARC.out"
    with pytest.raises(sparc_utils.SparcExecutionError):
        sparc_utils.assert_scf_converged(str(out), str(tmp_path))
