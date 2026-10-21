"""External-potential plugin for VASP QM/MM electrostatic embedding.

VASP imports this module and calls into it; never run it directly.
"""
from __future__ import annotations

import os
from pathlib import Path
import traceback
import warnings

import numpy as np

from pydft_qmmm.potentials.embedding.cell_mapping import read_mapping
from pydft_qmmm.potentials.embedding.grid_potential import EPS0
from pydft_qmmm.potentials.embedding.grid_potential import KJMOL_PER_EV
from pydft_qmmm.potentials.embedding.grid_potential import build_external_potential
from pydft_qmmm.potentials.embedding.grid_potential import contract_gaussian_gradient
from pydft_qmmm.potentials.embedding.grid_potential import electron_interaction_energy
from pydft_qmmm.potentials.embedding.grid_potential import electrostatic_potential_from_vasp
from pydft_qmmm.potentials.embedding.grid_potential import erfc_potential
from pydft_qmmm.potentials.embedding.grid_potential import gradient_at
from pydft_qmmm.potentials.embedding.grid_potential import interpolant_gradient_at
from pydft_qmmm.potentials.embedding.grid_potential import interpolate_at
from pydft_qmmm.potentials.embedding.grid_potential import interpolate_onto_grid
from pydft_qmmm.potentials.embedding.grid_potential import nuclear_charge_on_grid
from pydft_qmmm.potentials.embedding.grid_potential import poisson_fft
from pydft_qmmm.potentials.embedding.grid_potential import read_mm_charges
from pydft_qmmm.potentials.embedding.grid_potential import spectral_value_and_gradient
from pydft_qmmm.potentials.embedding.grid_potential import spline_value_and_gradient

__all__ = [
    "EPS0", "KJMOL_PER_EV", "CHARGE_FILE", "FORCE_FILE",
    "NET_FORCE_FILE", "PME_FORCE_FILE", "SENTINEL", "ENERGY_FILE",
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
PME_FORCE_FILE = "PME_FORCES"
SENTINEL = "PLUGIN_FIRED.txt"
ERROR_FILE = "PLUGIN_ERROR.txt"
PME_FILE = "PME_DATA"
# VASP's C++ plugin interface stores each callback's total_energy with
# cast<float>(), i.e. in single precision.  The exact values are logged
# here so that the driver can add back what the rounding removed.
ENERGY_FILE = "PLUGIN_ENERGY"

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
    mapping = read_mapping(shape, cell)
    metadata = Path("QM_CELL.json").read_bytes() if mapping is not None else b""
    key = (shape, cell.tobytes(), metadata)
    if mapping is not None and (erfc_near_enabled() or _coarse_near_gridnumber() is not None):
        raise ValueError("Separate cells require full-grid Gaussian embedding")
    if _CACHE.get("cell_key") == key and _CACHE.get("v_ext") is not None:
        return _CACHE["v_ext"]
    if os.path.isfile(PME_FILE):
        # PME path: the driver shipped the whole system and the Ewald
        # parameters, and helPME evaluates on VASP's own grid.
        from pydft_qmmm.potentials.embedding.pme_grid import build_pme_potential
        kwargs = {} if mapping is None else dict(source_cell=mapping.mm_cell, origin=mapping.origin)
        v_pme = build_pme_potential(PME_FILE, shape, cell, **kwargs)
        # In FFT mode PME contains only III; the near field supplies II
        # and its periodic images once.  In erfc mode II remains in PME
        # and the near field supplies its real-space complement.
        positions, charges, _, sigma = read_mm_charges(CHARGE_FILE)
        if erfc_near_enabled():
            # alpha must be the one the reciprocal sum used, or the two
            # halves do not add up to 1/r.  Take it from the same file
            # build_pme_potential read rather than from a second source.
            from pydft_qmmm.potentials.embedding.pme_grid import read_pme_data
            alpha = read_pme_data(PME_FILE)[3]
            v_near = erfc_potential(positions, charges, shape, cell, alpha)
        else:
            v_near = (_near_potential(positions, charges, shape, cell, sigma)
                      if mapping is None else mapping.restrict(build_external_potential(
                          positions, charges, mapping.mm_shape, mapping.mm_cell, sigma)))
        v_ext = v_pme + v_near
        _CACHE["cell_key"] = key
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
    v_ext = (_near_potential(positions, charges, shape, cell, sigma)
             if mapping is None else mapping.restrict(build_external_potential(
                 positions, charges, mapping.mm_shape, mapping.mm_cell, sigma)))
    _CACHE["cell_key"] = key
    _CACHE["shape"] = shape
    _CACHE["v_ext"] = v_ext
    with open(SENTINEL, "w") as fh:
        fh.write("local_potential callback executed\n")
    return v_ext


def _log_energy(kind, value):
    """Append one callback's exact energy addition (eV) to ENERGY_FILE."""
    with open(ENERGY_FILE, "a") as fh:
        fh.write(f"{kind} {float(value)!r}\n")


def _slab_data():
    import json
    path = Path("SLAB_DATA.json")
    return json.loads(path.read_text()) if path.exists() else None


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
        slab = _slab_data()
        if slab is not None:
            from pydft_qmmm.potentials.embedding.slab import grid_z
            z = grid_z(tuple(constants.shape_grid), np.asarray(constants.lattice_vectors),
                       slab["origin"])
            additions.total_potential += -2 * slab["coefficient"] * slab["moment"] * z
            if slab.get("qm_self", False):
                from pydft_qmmm.potentials.embedding.slab import qm_dipole
                cell = np.asarray(constants.lattice_vectors)
                positions = np.asarray(constants.positions) @ cell
                valence = np.asarray(constants.ZVAL)[np.asarray(constants.ion_types, dtype=int)]
                density = np.asarray(constants.charge_density)
                moment = qm_dipole(density, positions, valence, cell, slab["origin"])
                potential = -2 * slab["qm_coefficient"] * moment * z
                additions.total_potential += potential
                # The band energy includes integral(n*v). Replace it with C*Mq^2.
                additions.total_energy += (slab["qm_coefficient"] * moment**2
                                           - np.sum(density / density.size * potential))
        _log_energy("local_potential", additions.total_energy)
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
        mapping = read_mapping(shape, cell)
        positions = np.asarray(constants.positions) @ cell
        valence = np.asarray(constants.ZVAL, dtype=np.float64)[
            np.asarray(constants.ion_types, dtype=int)
        ]
        route = os.environ.get(NUCLEAR_FIELD_ENV, "spline").strip().lower()
        if route not in ("spline", "spectral"):
            raise ValueError(f"{NUCLEAR_FIELD_ENV} must be spline or spectral")
        if len(positions) > 0:
            v_ext = _external_potential(constants)
            # Differentiate the same interpolant used for the energy.
            # The spectral route remains a reference for grid convergence.
            evaluate = (spline_value_and_gradient if route == "spline"
                        else spectral_value_and_gradient)
            potential, gradient = evaluate(
                v_ext, cell, positions,
            )
            additions.total_energy += -float(np.sum(valence * potential))
            nuclear = valence[:, None] * gradient
            slab = _slab_data()
            if slab is not None:
                field = 2 * slab["coefficient"] * slab["moment"]
                additions.total_energy += field * np.dot(valence, positions[:, 2] + slab["origin"][2])
                nuclear[:, 2] -= field * valence
                if slab.get("qm_self", False):
                    from pydft_qmmm.potentials.embedding.slab import qm_dipole
                    moment = qm_dipole(constants.charge_density, positions, valence,
                                       cell, slab["origin"])
                    nuclear[:, 2] -= 2 * slab["qm_coefficient"] * moment * valence

            additions.forces += nuclear
            # VASP DESTROYS the net force on the QM ions.
            net = (
                np.asarray(constants.forces, dtype=np.float64).sum(axis=0)
                + nuclear.sum(axis=0)
            )
            with open(NET_FORCE_FILE, "w") as fh:
                fh.write(f"{len(positions)}\n")
                fh.write(f"{net[0]:.12e} {net[1]:.12e} {net[2]:.12e}\n")
        # Differentiate the discrete electronic and nuclear embedding energy.
        # VASP supplies N * electron charge per grid point in charge_density.
        # Use the final force-callback density, not a cached SCF input field.
        # Nuclear charges must transpose the SAME interpolant as -sum(Z*V).
        mm_positions, mm_charges, step, sigma = read_mm_charges(CHARGE_FILE)
        use_pme = os.path.isfile(PME_FILE)
        charge = None
        if len(mm_charges) or use_pme:
            density = np.asarray(constants.charge_density, dtype=np.float64)
            if density.shape != shape or not np.isfinite(density).all():
                raise ValueError("Invalid final VASP charge density")
            charge = nuclear_charge_on_grid(shape, cell, positions, valence, route)
            charge -= density / np.prod(shape)
        slab = _slab_data()
        if slab is not None:
            from pydft_qmmm.potentials.embedding.slab import qm_dipole, QM_NEUTRALITY_TOLERANCE, QM_NEUTRALITY_RELATIVE
            if slab["step"] != step:
                raise ValueError("Stale slab metadata")
            import json
            electron_profile = np.asarray(constants.charge_density).sum(axis=(0, 1)) / np.prod(shape)
            diagnostic = dict(step=step, neutrality_tolerance_e=max(QM_NEUTRALITY_TOLERANCE, QM_NEUTRALITY_RELATIVE * float(valence.sum())),
                              charge_residual_e=float(valence.sum()-electron_profile.sum()), electron_count=float(electron_profile.sum()),
                              ionic_charge=float(valence.sum()), origin=slab["origin"],
                              electron_count_by_z=electron_profile.tolist(),
                              z_spacing_A=float(cell[2, 2]/shape[2]))
            Path("SLAB_DIAGNOSTICS.json").write_text(json.dumps(diagnostic, indent=2)+"\n")
            moment = qm_dipole(constants.charge_density, positions, valence,
                               cell, slab["origin"])
            reaction = np.zeros((len(slab["charges"]), 3))
            reaction[:, 2] = (-2 * slab["coefficient"] * moment
                              * np.asarray(slab["charges"]) * KJMOL_PER_EV)
            with open("SLAB_FORCES", "w") as fh:
                fh.write(f"{len(reaction)} {step}\n")
                np.savetxt(fh, reaction, fmt="%.12e")
        mm_forces = np.zeros((len(mm_charges), 3))
        if len(mm_charges):
            if erfc_near_enabled() or _coarse_near_gridnumber() is not None:
                raise ValueError("Consistent MM forces require the full-grid Gaussian near field")
            near_shape = shape if mapping is None else mapping.mm_shape
            near_cell = cell if mapping is None else mapping.mm_cell
            near_charge = charge if mapping is None else mapping.transpose(charge)
            rho = near_charge * np.prod(near_shape) / abs(np.linalg.det(near_cell))
            phi_qm = poisson_fft(rho, near_cell)
            mm_forces = contract_gaussian_gradient(
                phi_qm, mm_positions, mm_charges, near_shape, near_cell, sigma,
            ) * KJMOL_PER_EV
        with open(FORCE_FILE, "w") as fh:
            fh.write(f"{len(mm_forces)} {step}\n")
            for fx, fy, fz in mm_forces:
                fh.write(f"{fx:.12e} {fy:.12e} {fz:.12e}\n")
        # The reciprocal back-reaction on the PME_DATA sources, from the
        # same grid charge: the transpose of build_pme_potential.
        if use_pme:
            from pydft_qmmm.potentials.embedding.pme_grid import pme_source_forces
            kwargs = {} if mapping is None else dict(source_cell=mapping.mm_cell, origin=mapping.origin)
            pme_forces = pme_source_forces(PME_FILE, charge, shape, cell, **kwargs)
            with open(PME_FORCE_FILE, "w") as fh:
                fh.write(f"{len(pme_forces)} {step}\n")
                for fx, fy, fz in pme_forces:
                    fh.write(f"{fx:.12e} {fy:.12e} {fz:.12e}\n")
        _log_energy("force_and_stress", additions.total_energy)
    except Exception as exc:
        _record_error(exc)
        raise
