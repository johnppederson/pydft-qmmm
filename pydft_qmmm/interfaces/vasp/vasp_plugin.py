"""External-potential plugin for VASP QM/MM electrostatic embedding.

VASP imports this module and calls into it; never run it directly.
"""
from __future__ import annotations

import os
import traceback
import warnings

import numpy as np

from pydft_qmmm.embedding.grid_potential import EPS0
from pydft_qmmm.embedding.grid_potential import KJMOL_PER_EV
from pydft_qmmm.embedding.grid_potential import build_external_potential
from pydft_qmmm.embedding.grid_potential import contract_gaussian_gradient
from pydft_qmmm.embedding.grid_potential import electron_interaction_energy
from pydft_qmmm.embedding.grid_potential import electrostatic_potential_from_vasp
from pydft_qmmm.embedding.grid_potential import erfc_potential
from pydft_qmmm.embedding.grid_potential import gradient_at
from pydft_qmmm.embedding.grid_potential import interpolant_gradient_at
from pydft_qmmm.embedding.grid_potential import interpolate_at
from pydft_qmmm.embedding.grid_potential import interpolate_onto_grid
from pydft_qmmm.embedding.grid_potential import read_mm_charges
from pydft_qmmm.embedding.grid_potential import spectral_value_and_gradient
from pydft_qmmm.embedding.grid_potential import spline_value_and_gradient

__all__ = [
    "EPS0", "KJMOL_PER_EV", "CHARGE_FILE", "FORCE_FILE",
    "NET_FORCE_FILE", "SENTINEL",
    "ERROR_FILE",
    "build_external_potential", "contract_gaussian_gradient",
    "electron_interaction_energy", "electrostatic_potential_from_vasp",
    "erfc_potential", "erfc_near_enabled",
    "gradient_at", "interpolant_gradient_at", "interpolate_at",
    "interpolate_onto_grid",
    "read_mm_charges", "spectral_value_and_gradient",
    "reset_cache", "local_potential", "force_and_stress",
]

CHARGE_FILE = "MM_CHARGES"
FORCE_FILE = "MM_FORCES"
NET_FORCE_FILE = "QM_NET_FORCE"
SENTINEL = "PLUGIN_FIRED.txt"
ERROR_FILE = "PLUGIN_ERROR.txt"
PME_FILE = "PME_DATA"

# Opt-in: build the analytic near field on an N**3 grid and interpolate
# it onto VASP's.
COARSE_NEAR_ENV = "PYDFT_QMMM_VASP_COARSE_NEAR"

# Opt-in: reinstate the near field as the Ewald real-space correction.
ERFC_NEAR_ENV = "PYDFT_QMMM_VASP_ERFC_NEAR"

# Keep the exact Fourier evaluation available for convergence comparisons.
NUCLEAR_FIELD_ENV = "PYDFT_QMMM_VASP_NUCLEAR_FIELD"

_CACHE = {}


def _coarse_near_gridnumber():
    """The coarse grid to build v_near on, or None for the normal path.

    Returns:
        The requested grid number, or None if the variable is unset or
        empty.

    Raises:
        ValueError: If the variable is set to something that is not a
            positive integer.
    """
    raw = os.environ.get(COARSE_NEAR_ENV, "").strip()
    if not raw:
        return None
    try:
        gridnumber = int(raw)
    except ValueError:
        raise ValueError(
            f"{COARSE_NEAR_ENV}={raw!r} is not an integer",
        ) from None
    if gridnumber < 1:
        raise ValueError(f"{COARSE_NEAR_ENV}={raw!r} must be positive")
    return gridnumber


def erfc_near_enabled():
    """Whether the real-space route is selected.

    Read on every call rather than cached at import, and shared with the
    driver, so that one variable cannot switch the two halves apart.
    """
    return bool(os.environ.get(ERFC_NEAR_ENV, "").strip())


def _near_potential(positions, charges, shape, cell, sigma):
    """Return v_near on VASP's grid by the selected route."""
    gridnumber = _coarse_near_gridnumber()
    if gridnumber is None:
        return build_external_potential(
            positions, charges, shape, cell, sigma,
        )
    coarse = build_external_potential(
        positions, charges, (gridnumber,) * 3, cell, sigma,
    )
    return interpolate_onto_grid(coarse, shape)


def reset_cache():
    """Discard the cached external potential.  For tests."""
    _CACHE.clear()


def _record_error(exc):
    """Persist a traceback before re-raising.

    VASP wraps the callback in a bare ``except`` (see
    ``src/plugins/src/vasp/_apply_interface.py``) and turns any
    exception into a PYTHON_EXCEPTION return code.
    """
    with open(ERROR_FILE, "w") as fh:
        fh.write("".join(traceback.format_exception(exc)))


def _external_potential(constants):
    """Return V_ext on the current grid, building it at most once.

    The MM geometry is fixed for the duration of one VASP launch, so the
    field is built on the first callback and reused for every subsequent
    SCF step.  The cache is keyed on the grid shape so that a changed
    grid rebuilds rather than returning a mismatched array.
    """
    shape = tuple(int(n) for n in constants.shape_grid)
    cell = np.asarray(constants.lattice_vectors, dtype=np.float64)
    if _CACHE.get("shape") == shape and _CACHE.get("v_ext") is not None:
        return _CACHE["v_ext"]
    if os.path.isfile(PME_FILE):
        # PME path: the driver shipped the whole system and the Ewald
        # parameters, and helPME evaluates on VASP's own grid.
        from pydft_qmmm.embedding.pme_grid import build_pme_potential
        v_pme = build_pme_potential(PME_FILE, shape, cell)
        # In FFT mode PME contains only III; the near field supplies II
        # and its periodic images once.  In erfc mode II remains in PME
        # and the near field supplies its real-space complement.
        positions, charges, _, sigma = read_mm_charges(CHARGE_FILE)
        if erfc_near_enabled():
            # alpha must be the one the reciprocal sum used, or the two
            # halves do not add up to 1/r.  Take it from the same file
            # build_pme_potential read rather than from a second source.
            from pydft_qmmm.embedding.pme_grid import read_pme_data
            alpha = read_pme_data(PME_FILE)[3]
            v_near = erfc_potential(positions, charges, shape, cell, alpha)
        else:
            v_near = _near_potential(
                positions, charges, shape, cell, sigma,
            )
        v_ext = v_pme + v_near
        _CACHE["shape"] = shape
        _CACHE["v_ext"] = v_ext
        with open(SENTINEL, "w") as fh:
            fh.write("local_potential callback executed (PME + analytic)\n")
        return v_ext
    positions, charges, _, sigma = read_mm_charges(CHARGE_FILE)
    net_charge = float(np.sum(charges)) if len(charges) else 0.0
    if abs(net_charge) > 1e-6:
        # The G=0 term is dropped in the Poisson solve, which is
        # equivalent to a uniform neutralizing background.
        warnings.warn(
            f"subsystem II carries a net charge of {net_charge:.4f} e.  "
            "Absolute embedded energies are shifted by a constant from "
            "the neutralizing-background (G=0) convention.",
            RuntimeWarning,
            stacklevel=2,
        )
    v_ext = _near_potential(positions, charges, shape, cell, sigma)
    _CACHE["shape"] = shape
    _CACHE["v_ext"] = v_ext
    with open(SENTINEL, "w") as fh:
        fh.write("local_potential callback executed\n")
    return v_ext


def local_potential(constants, additions):
    """Add the MM external potential to VASP's local potential.

    Defines the PLUGINS/LOCAL_POTENTIAL interface.  Called once per SCF
    step.
    """
    try:
        if getattr(constants, "hartree_potential", None) is not None:
            _CACHE["hartree"] = np.array(constants.hartree_potential)
            _CACHE["ion"] = np.array(constants.ion_potential)
        additions.total_potential += _external_potential(constants)
    except Exception as exc:
        _record_error(exc)
        raise


def force_and_stress(constants, additions):
    """Apply the nuclear terms VASP omits.

    Defines the PLUGINS/FORCE_AND_STRESS interface.  VASP does not
    include the interaction between the external potential and the
    pseudo-ion cores, so add

        dE_I = -Z_I V_ext(R_I)
        dF_I = +Z_I grad V_ext(R_I)

    Z_I is the pseudopotential valence charge ZVAL, NOT the atomic
    number: VASP's nuclei are pseudo-ions carrying only valence charge,
    so the atomic number would over-count by the core electrons (79 vs
    11 for gold).
    """
    try:
        cell = np.asarray(constants.lattice_vectors, dtype=np.float64)
        shape = tuple(int(n) for n in constants.shape_grid)
        positions = np.asarray(constants.positions) @ cell
        if len(positions) > 0:
            v_ext = _external_potential(constants)
            valence = np.asarray(constants.ZVAL, dtype=np.float64)[
                np.asarray(constants.ion_types, dtype=int)
            ]
            # Differentiate the same interpolant used for the energy.
            # The spectral route remains a reference for grid convergence.
            route = os.environ.get(NUCLEAR_FIELD_ENV, "spline").strip().lower()
            if route not in ("spline", "spectral"):
                raise ValueError(f"{NUCLEAR_FIELD_ENV} must be spline or spectral")
            evaluate = (spline_value_and_gradient if route == "spline"
                        else spectral_value_and_gradient)
            potential, gradient = evaluate(
                v_ext, cell, positions,
            )
            additions.total_energy += -float(np.sum(valence * potential))
            nuclear = valence[:, None] * gradient
            additions.forces += nuclear
            # VASP DESTROYS the net force on the QM ions.
            net = (
                np.asarray(constants.forces, dtype=np.float64).sum(axis=0)
                + nuclear.sum(axis=0)
            )
            with open(NET_FORCE_FILE, "w") as fh:
                fh.write(f"{len(positions)}\n")
                fh.write(f"{net[0]:.12e} {net[1]:.12e} {net[2]:.12e}\n")
        # The QM->MM back-reaction.
        phi_qm = electrostatic_potential_from_vasp(
            _CACHE.get("hartree"), _CACHE.get("ion"),
        )
        mm_positions, mm_charges, step, sigma = read_mm_charges(CHARGE_FILE)
        mm_forces = contract_gaussian_gradient(
            phi_qm, mm_positions, mm_charges, shape, cell, sigma,
        ) * KJMOL_PER_EV
        with open(FORCE_FILE, "w") as fh:
            fh.write(f"{len(mm_forces)} {step}\n")
            for fx, fy, fz in mm_forces:
                fh.write(f"{fx:.12e} {fy:.12e} {fz:.12e}\n")
    except Exception as exc:
        _record_error(exc)
        raise
