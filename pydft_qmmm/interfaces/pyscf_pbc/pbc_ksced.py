"""Gamma-point, monomolecular KSCED with one fixed frozen environment."""
from __future__ import annotations

__all__ = ["KSCEDPBCPotential"]

from dataclasses import dataclass, field, replace
from numbers import Integral
from types import MappingProxyType
from typing import Any

import numpy as np

from pydft_qmmm.interfaces import ElectrostaticCouplingMode
from pydft_qmmm.utils import BOHR_PER_ANGSTROM, KJMOL_PER_EH, system_cache
from ..pyscf.pyscf_backend import load_backend, load_submodule, to_like, to_numpy
from .pbc_cell import build_cell, valence_charges
from .pbc_interface import PySCFPBCPotential, _SCFState, _nuclear_coupling
from .pbc_embedding import grid_coordinates, external_potential, near_potential, ao_operator


@dataclass(frozen=True)
class _KSCEDState:
    cell: Any
    qm_indices: tuple[int, ...]
    method: Any
    dm: Any
    embedding_state: Any = None
    nuclear_energy: float = 0.0
    electronic_energy: float = 0.0


def _selection(system: Any, value: Any, label: str) -> tuple[int, ...]:
    if isinstance(value, str):
        value = system.select(value)
    try:
        indices = tuple(value)
    except TypeError as error:
        raise ValueError(f"KSCED {label} must be a selection or atom indices") from error
    if (not indices or any(not isinstance(i, Integral) or isinstance(i, bool)
                           for i in indices)
            or len(set(indices)) != len(indices)
            or any(i < 0 or i >= len(system.positions) for i in indices)):
        raise ValueError(f"KSCED {label} must contain unique, nonempty, inbounds indices")
    return tuple(sorted(int(i) for i in indices))


def _functional(value: Any, label: str) -> None:
    from pyscf.dft import libxc
    try:
        valid = (isinstance(value, str) and bool(value)
                 and libxc.xc_type(value) in ("LDA", "GGA")
                 and not libxc.is_hybrid_xc(value)
                 and not libxc.is_nlc(value))
    except (ValueError, KeyError, RuntimeError) as error:
        raise ValueError(f"KSCED {label} must be a pure LDA/GGA functional") from error
    if not valid:
        raise ValueError(f"KSCED {label} must be a pure LDA/GGA functional")


@dataclass(frozen=True)
class KSCEDPBCPotential(PySCFPBCPotential):
    """Embed active QM atoms in a density converged once on frozen QM atoms.

    The outer charge and multiplicity belong to A, which is subsystem I.
    Frozen B remains in MM for its interactions with the rest of the system.
    Energies are E_ainb (without B's self energy); forces are
    the analytic A gradient and A–C reaction forces, with zero rows on B.
    The near field uses periodic Gaussian charges, as in the plain PBC backend.
    """
    ksced: dict[str, Any] = field(default_factory=dict)
    active_indices: tuple[int, ...] = field(default=(), init=False)
    frozen_indices: tuple[int, ...] = field(default=(), init=False)
    frozen_method: list[Any] = field(default_factory=lambda: [None], init=False)
    frozen_environment: list[Any] = field(default_factory=lambda: [None], init=False)
    _fixed: list[Any] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.ksced, dict):
            raise ValueError("ksced must be a dictionary")
        config = dict(self.ksced)
        allowed = {"active_atoms", "frozen_atoms", "basis_mode", "t_nad",
                   "frozen_charge", "frozen_multiplicity"}
        unknown = set(config) - allowed
        if unknown:
            raise ValueError(f"unknown KSCED settings: {sorted(unknown)}")
        if not {"active_atoms", "frozen_atoms"} <= set(config):
            raise ValueError("KSCED requires active_atoms and frozen_atoms")
        config.setdefault("basis_mode", "M")
        config.setdefault("t_nad", "LDA_K_TF")
        config.setdefault("frozen_charge", 0)
        config.setdefault("frozen_multiplicity", 1)
        if config["basis_mode"] != "M":
            raise ValueError("KSCED supports only basis_mode='M'")
        if self.multiplicity < 1 or config["frozen_multiplicity"] < 1:
            raise ValueError("KSCED multiplicities must be positive")
        _functional(self.functional, "functional")
        _functional(config["t_nad"], "t_nad")
        if not isinstance(self.pseudo, str) or not self.pseudo.lower().startswith("gth-"):
            raise ValueError("KSCED requires a GTH pseudopotential")
        for name in ("active_atoms", "frozen_atoms"):
            config[name] = _selection(self.system, config[name], name)
        active, frozen = config["active_atoms"], config["frozen_atoms"]
        if set(active) & set(frozen) or set(active) != set(
                self.system.select("subsystem I")):
            raise ValueError("KSCED active atoms must equal subsystem I and be disjoint from frozen B")
        object.__setattr__(self, "ksced", MappingProxyType(config))
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))
        if self.mesh is not None:
            object.__setattr__(self, "mesh", tuple(self.mesh))
        object.__setattr__(self, "active_indices", active)
        object.__setattr__(self, "frozen_indices", frozen)
        self._validate_options()
        self._fixed.extend([
            np.array(self.system.positions[list(frozen)], copy=True),
            tuple(str(self.system.elements[i]) for i in sorted(active + frozen)),
            np.array(self.system.box, copy=True),
        ])

    def _validate_options(self) -> None:
        # These options would change the Gamma/FFT model or fixed-N ensemble.
        unsupported = {"kpt", "kpts", "with_df", "grids", "density_fit",
                       "cell", "mol", "xc", "nlc", "omega", "mu0", "fix_mu"}
        if unsupported.intersection(self.options):
            raise ValueError("KSCED requires Gamma RKS/UKS, FFTDF and fixed electron number")
        sigma = self.options.get("sigma", 0)
        if not isinstance(sigma, (int, float)) or not np.isfinite(sigma) or sigma < 0:
            raise ValueError("KSCED sigma must be a finite nonnegative number")
        if self.options.get("smearing_method", "fermi") not in ("fermi", "gaussian"):
            raise ValueError("KSCED smearing_method must be 'fermi' or 'gaussian'")
        if "smearing_method" in self.options and not sigma:
            raise ValueError("KSCED smearing_method requires positive sigma")

    def _validate_frame(self) -> None:
        if self.potentials:
            from .pbc_pme import PeriodicPMEElectronicPotential
            if any(not isinstance(p, PeriodicPMEElectronicPotential)
                   or p.excluded_indices != self.frozen_indices for p in self.potentials):
                raise ValueError("KSCED requires PME potentials that exclude frozen B")
        active, frozen = self.active_indices, self.frozen_indices
        if set(active) != set(self.system.select("subsystem I")):
            raise ValueError("KSCED requires a fixed QM partition")
        if tuple(str(self.system.elements[i]) for i in sorted(active + frozen)) != self._fixed[1]:
            raise ValueError("KSCED requires fixed active and frozen elements")
        if not np.array_equal(self.system.positions[list(frozen)], self._fixed[0]):
            raise ValueError("KSCED frozen atom coordinates changed")
        if not np.array_equal(self.system.box, self._fixed[2]):
            raise ValueError("KSCED requires a fixed box/lattice")

    def electrostatic_coupling_mode(self) -> ElectrostaticCouplingMode:
        return ElectrostaticCouplingMode.ENGINE

    def configure_electrostatic_embedding(self, enabled: bool) -> None:
        if self.density_guess[0] is not None and enabled != self.embedding:
            raise ValueError("Configure KSCED embedding before the first SCF")
        super().configure_electrostatic_embedding(enabled)
        self._validate_frame()

    def add_electronic_potential(self, potential: Any) -> None:
        if self.density_guess[0] is not None:
            raise ValueError("Add KSCED potentials before the first SCF")
        # Arbitrary fields cannot guarantee frozen-B source exclusions or
        # conservative source forces. Accept only the supported PME field.
        if not hasattr(potential, "pme"):
            raise ValueError("KSCED supports only periodic PME added potentials")
        from pydft_qmmm.potentials.pme_potential import PMEElectronicPotential
        from .pbc_pme import PeriodicPMEElectronicPotential
        if not isinstance(potential, PMEElectronicPotential):
            raise ValueError("KSCED supports only periodic PME added potentials")
        if potential.system is not self.system:
            raise ValueError("KSCED PME potential must use the same system")
        self.potentials.append(PeriodicPMEElectronicPotential(
            potential.system, potential.pme_alpha, potential.pme_gridnumber,
            potential.pme_spline_order, excluded_indices=self.frozen_indices,
        ))

    def _cell(self, indices: tuple[int, ...], charge: int, multiplicity: int) -> Any:
        return build_cell(self.system, self.basis, self.pseudo, self.ke_cutoff,
                          self.mesh, charge, multiplicity, self.verbose, indices)[0]

    def _method(self, cell: Any, multiplicity: int, active: bool) -> Any:
        backend = load_backend(self.device)
        dft = load_submodule(backend, "pbc.dft")
        solver = dft.UKS if multiplicity > 1 else dft.RKS
        method = solver(cell, xc=self.functional)
        method.with_df = load_submodule(backend, "pbc.df").FFTDF(cell)
        method.grids = load_submodule(backend, "pbc.dft.gen_grid").UniformGrids(cell)
        method.conv_tol, method.max_cycle = self.conv_tol, self.max_cycle
        for key, value in self.options.items():
            if key in ("sigma", "smearing_method"):
                continue
            setattr(method, key, value)
        if active and self.options.get("sigma", 0):
            method = method.smearing(sigma=self.options["sigma"],
                                     method=self.options.get("smearing_method", "fermi"))
        return method

    def nuclear_charges(self) -> Any:
        self._validate_frame()
        return valence_charges(self._cell(
            self.active_indices, self.charge, self.multiplicity,
        ))

    def _scf_state(self) -> _KSCEDState:
        if self.potentials and not self.embedding:
            raise ValueError("Enable KSCED electrostatic embedding before using PME potentials")
        self._validate_frame()
        return self._cached_ksced_state()

    @system_cache("positions", "charges", "elements", "subsystems", "box")
    def _cached_ksced_state(self) -> _KSCEDState:
        try:
            from pyscf import ksced
        except ImportError as error:
            raise ImportError("KSCED requires the optional pyscf-ksced plugin") from error
        cell = self._cell(self.active_indices, self.charge, self.multiplicity)
        mf_a = self._method(cell, self.multiplicity, True)
        if self.frozen_method[0] is None:
            cell_b = self._cell(self.frozen_indices, self.ksced["frozen_charge"],
                                self.ksced["frozen_multiplicity"])
            if not np.array_equal(cell.mesh, cell_b.mesh):
                raise ValueError("KSCED A/B cutoff meshes differ; supply a common explicit mesh")
            mf_b = self._method(cell_b, self.ksced["frozen_multiplicity"], False)
            mf_b.kernel()
            if not mf_b.converged:
                raise RuntimeError("KSCED frozen B SCF did not converge")
            environment = ksced.frozen_env(mf_b, cell)
            self.frozen_method[0] = mf_b
            self.frozen_environment[0] = environment
        mf_ainb = ksced.embed(mf_a, self.frozen_method[0], basis_mode="M",
                             env=self.frozen_environment[0])
        mf_ainb.t_nad = self.ksced["t_nad"]
        embedding_state = None
        nuclear_energy = 0.0
        if self.embedding:
            backend = load_backend(self.device)
            kpts = np.zeros((1, 3))
            coords, weights = grid_coordinates(cell)
            indices = tuple(sorted(set(self.system.select("subsystem II"))
                                   - set(self.frozen_indices)))
            potential = external_potential(self.system, cell, self.potentials,
                                           indices, self.embedding_sigma)
            matrix = ao_operator(backend, cell, kpts, coords, weights, potential)
            # Install AFTER KSCED wrapping. _vne_a uses super().get_hcore,
            # so frozen B never couples to this MM field.
            core = mf_ainb.get_hcore()
            shifted = core + to_like(matrix[0].real, core)
            mf_ainb.get_hcore = lambda *args, **kwargs: shifted
            nuclear_energy = _nuclear_coupling(cell, potential, self.system.box)
            embedding_state = _SCFState(
                cell, self.active_indices, indices, kpts, mf_ainb, None,
                coords, weights, potential, matrix, nuclear_energy,
                tuple(self.potentials),
            )
        self.frame[0] += 1
        stream = None
        if self.output_file is not None and self.frame[0] % self.output_interval == 0:
            stream = open(self.output_file, "a")
            mf_ainb.stdout, mf_ainb.verbose = stream, max(self.verbose, 4)
        try:
            mf_ainb.kernel(dm0=self.density_guess[0])
        finally:
            if stream is not None:
                mf_ainb.stdout = cell.stdout
                stream.close()
        if not mf_ainb.converged:
            raise RuntimeError("KSCED active A SCF did not converge")
        dm = mf_ainb.make_rdm1()
        self.density_guess[0] = dm
        electronic_energy = 0.0
        if embedding_state is not None:
            # Shared PBC embedding contracts one spin-summed matrix per k-point.
            total_dm = np.asarray(to_numpy(dm))
            if total_dm.ndim == 3:
                total_dm = total_dm.sum(axis=0)
            embedding_state = replace(embedding_state, dm=total_dm[None])
            electronic_energy = float(np.einsum(
                "ij,ji->", total_dm, embedding_state.matrix[0],
            ).real)
        return _KSCEDState(cell, self.active_indices, mf_ainb, dm,
                           embedding_state, nuclear_energy, electronic_energy)

    def compute_energy(self) -> float:
        state = self._scf_state()
        return (float(state.method.energy_potential()) + state.nuclear_energy) * KJMOL_PER_EH

    def compute_components(self) -> dict[str, float]:
        """Return additive energy components in kJ/mol, at the embedded density.

        B's self energy is omitted. B-C terms belong to the MM calculator.
        Validate even on a cache hit, and do not expose the cached dictionary.
        """
        self._scf_state()
        return dict(self._cached_components())

    @system_cache("positions", "charges", "elements", "subsystems", "box")
    def _cached_components(self) -> dict[str, float]:
        state = self._scf_state()
        mf, cell = state.method, state.cell
        summary = {key: float(to_numpy(value))
                   for key, value in mf.scf_summary.items()}
        total_dm = np.asarray(to_numpy(state.dm))
        if total_dm.ndim == 3:
            total_dm = total_dm.sum(axis=0)
        # Evaluate A's XC at its embedded density, without another SCF.
        ni = mf._numint
        nr = ni.nr_uks if state.dm.ndim == 3 else ni.nr_rks
        exc_a = float(to_numpy(nr(
            cell, mf.grids, mf.xc, state.dm, 0, 1, mf.kpt, None,
            max_memory=mf.max_memory,
        )[1]))
        v_b = np.asarray(to_numpy(mf.with_env.get_vne_b(cell, mf.kpt)))
        a_electrons_b_nuclei = float(np.einsum("ij,ji->", total_dm, v_b).real)
        half_j_ab = summary["ksced_coul_ab_half"]
        t_nad = float(to_numpy(mf.e_tnad))
        nuclear_a = float(cell.energy_nuc())
        components = {
            "KSCED Active Energy": (
                summary["e1"] - state.electronic_energy - a_electrons_b_nuclei
                + summary["coul"] - half_j_ab + exc_a + nuclear_a
            ),
            "A-B Nuclear Interaction": float(mf.energy_nuc()) - nuclear_a,
            "A-B Electron-Nuclear Interaction": (
                a_electrons_b_nuclei + summary["ksced_vne_a_rho_b"]
            ),
            "A-B Hartree Interaction": 2 * half_j_ab,
            "Non-additive XC": summary["exc"] - t_nad - exc_a,
            "Non-additive Kinetic": t_nad,
            "Electronic Entropy Correction": float(mf.energy_potential()) - float(mf.e_tot),
            "A-C Near Electronic Embedding": 0.0,
            "A-C Near Nuclear Embedding": 0.0,
            "A-C Reciprocal Electronic Embedding": 0.0,
            "A-C Reciprocal Nuclear Embedding": 0.0,
        }
        embedded = state.embedding_state
        if embedded is not None:
            if not embedded.potentials:
                near_electronic = state.electronic_energy
                near_nuclear = state.nuclear_energy
            elif not embedded.embed_indices:
                near_electronic = near_nuclear = 0.0
            else:
                # Split the linear external operator at the SAME converged
                # density. Separate SCFs would include polarization twice.
                field = near_potential(
                    self.system, embedded.embed_indices, tuple(cell.mesh),
                    np.asarray(self.system.box), self.embedding_sigma,
                ).reshape(-1)
                matrix = ao_operator(
                    load_backend(self.device), cell, embedded.kpts,
                    embedded.coords, embedded.weights, field,
                )[0]
                near_electronic = float(np.einsum("ij,ji->", total_dm, matrix).real)
                near_nuclear = _nuclear_coupling(cell, field, self.system.box)
            components.update({
                "A-C Near Electronic Embedding": near_electronic,
                "A-C Near Nuclear Embedding": near_nuclear,
                "A-C Reciprocal Electronic Embedding": state.electronic_energy - near_electronic,
                "A-C Reciprocal Nuclear Embedding": state.nuclear_energy - near_nuclear,
            })
        return {key: value * KJMOL_PER_EH for key, value in components.items()}

    def compute_forces(self) -> Any:
        from .pbc_forces import pulay_forces, nuclear_forces, mm_forces
        state = self._scf_state()
        # The plugin differentiates the native A/B pseudopotentials and checks
        # the native hcore on undo_ksced(). External AO derivatives belong here.
        # Retain converged orbitals/eigenvalues: their overlap Pulay term already
        # contains the external field. Restore the SCF operator even on failure.
        shifted = state.method.__dict__.pop("get_hcore", None)
        try:
            gradient = np.asarray(to_numpy(state.method.nuc_grad_method().kernel()))
        finally:
            if shifted is not None:
                state.method.get_hcore = shifted
        forces = np.zeros(np.asarray(self.system.positions).shape, dtype=float)
        forces[list(self.active_indices)] = -gradient
        if state.embedding_state is not None:
            backend = load_backend(self.device)
            embedded = state.embedding_state
            forces += pulay_forces(backend, embedded, len(forces))
            forces += nuclear_forces(embedded, self.system.box, len(forces))
            forces += mm_forces(backend, embedded, self.system,
                                self.embedding_sigma, len(forces))
        return forces * KJMOL_PER_EH * BOHR_PER_ANGSTROM
