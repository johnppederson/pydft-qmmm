"""The VASP software interface and potential.

This module wraps VASP as a QM potential within PyDFT-QMMM.  Each energy
or force evaluation writes a fresh set of VASP input files into a
persistent working directory.
"""
from __future__ import annotations

__all__ = ["VaspInterface", "VaspPotential"]

import os
from dataclasses import dataclass
from dataclasses import field
from typing import ClassVar
from typing import TYPE_CHECKING

import numpy as np

from pydft_qmmm.interfaces import ElectrostaticCouplingMode
from pydft_qmmm.interfaces import QMInterface
from pydft_qmmm.interfaces.engine_embedding import EngineEmbeddingMixin
from pydft_qmmm.potentials import AtomicPotential
from pydft_qmmm.utils import KJMOL_PER_EV
from pydft_qmmm.utils import system_cache

from . import vasp_utils

if TYPE_CHECKING:
    from typing import Any
    from numpy.typing import NDArray
    from pydft_qmmm.potentials import ElectronicPotential
    from pydft_qmmm import System  # noqa: F401


@dataclass(frozen=True)
class VaspInterface(EngineEmbeddingMixin, QMInterface):
    r"""A mix-in for storing and manipulating VASP data types.

    Args:
        system: The system that will inform the interface to VASP.
        charge: The net charge (:math:`e`) of the QM subsystem.  VASP
            has no molecular charge concept, so a non-zero charge must
            be realized by setting NELECT together with a compensating
            background; only ``charge=0`` is currently accepted.
        directory: The working directory for VASP calculations.
        command: The shell command that launches VASP.
        incar: INCAR tags applied to every calculation.
        kpts: The number of k-points along each reciprocal lattice
            vector.
        pp_path: The directory containing per-species POTCAR
            subdirectories.
        potcar_map: A mapping from element symbol to POTCAR
            subdirectory name, for selecting non-default potentials.

    Attributes:
        potentials: A list of electronic potentials to incorporate into
            QM calculations.
        frame: The number of calculations performed so far, used to
            decide whether a restart file is available.
    """
    _embedding_conflict_message: ClassVar[str] = (
        "VASP embedding=True conflicts with this QMMMHamiltonian: "
        "no QM/MM electrostatic interaction is assigned to the QM "
        "level. Disable embedding or select electrostatic coupling "
        "to avoid double-counting electrostatics."
    )
    _embedding_unavailable_message: ClassVar[str] = (
        "PME embedding needs the VASP Python plugin.  Build the "
        "potential with embedding=True and a vasp_std compiled "
        "with -DPLUGINS."
    )

    charge: int
    directory: str
    command: str
    incar: dict[str, Any]
    kpts: tuple[int, int, int]
    pp_path: str
    potcar_map: dict[str, str]
    embedding: bool = False
    embedding_sigma: float = 0.3
    potentials: list[ElectronicPotential] = field(
        default_factory=list,
        init=False,
    )
    frame: list[int] = field(
        default_factory=lambda: [0],
        init=False,
    )

    def electrostatic_coupling_mode(self) -> ElectrostaticCouplingMode:
        """Keep electrostatic coupling inside the VASP plugin."""
        return ElectrostaticCouplingMode.ENGINE

    def applies_nuclear_potential(self) -> bool:
        """The plugin couples V_ext to the QM nuclei itself.

        ``vasp_plugin.force_and_stress`` adds ``-sum(ZVAL * V_ext)`` and
        ``+ZVAL * grad V_ext`` for every ion. So the coupling Hamiltonian
        must not add its own nuclear term on top.

        Returns:
            Whether the nuclear term is already applied, i.e. whether
            electrostatic embedding is switched on.
        """
        return self.embedding

    def _write_input(self) -> list[int]:
        """Write the VASP input files for the current QM geometry.

        Returns:
            The permutation mapping POSCAR order onto the sorted QM
            atom indices.
        """
        if self.embedding and int(self.incar.get("ISYM", 0)) > 0:
            raise ValueError(
                "ISYM must be 0 under electrostatic embedding.",
            )
        os.makedirs(self.directory, exist_ok=True)
        qm_indices = sorted(self.system.select("subsystem I"))
        symbols = [str(self.system.elements[i]) for i in qm_indices]
        positions = np.asarray(self.system.positions)[qm_indices]
        # PyDFT-QMMM stores lattice vectors as columns; VASP wants rows.
        cell = np.asarray(self.system.box).T
        species, _, order = vasp_utils.write_poscar(
            os.path.join(self.directory, "POSCAR"),
            symbols,
            positions,
            cell,
        )
        vasp_utils.write_potcar(
            os.path.join(self.directory, "POTCAR"),
            species,
            self.pp_path,
            self.potcar_map,
        )
        vasp_utils.write_kpoints(
            os.path.join(self.directory, "KPOINTS"),
            self.kpts,
        )
        tags = dict(self.incar)
        # Single point only; PyDFT-QMMM owns the dynamics.
        tags.update({"NSW": 0, "IBRION": -1})
        if self.embedding:
            # PLUGINS/MODE is deliberately left unset.  It defaults to
            # "serial", which is what makes this correct under MPI: only
            # rank 1 calls the plugin, it receives the FULL gathered
            # grid, and the addition is broadcast back.  Under
            # PLUGINS/MODE = parallel every rank would be handed its own
            # grid slab and a full-grid V_ext would be silently wrong.
            tags["PLUGINS/LOCAL_POTENTIAL"] = "T"
            tags["PLUGINS/FORCE_AND_STRESS"] = "T"
            # VASP looks for the plugin with find_spec("vasp_plugin")
            with open(
                os.path.join(self.directory, "vasp_plugin.py"), "w",
            ) as fh:
                fh.write(
                    '"""Generated by VaspInterface._write_input.\n\n'
                    "VASP imports this by bare name from the launch\n"
                    "directory.  The implementation lives in the\n"
                    'installed package.\n"""\n'
                    "from pydft_qmmm.interfaces.vasp.vasp_plugin import (\n"
                    "    force_and_stress,\n"
                    "    local_potential,\n"
                    ")\n\n"
                    '__all__ = ["local_potential", "force_and_stress"]\n',
                )
            # The plugin writes MM_FORCES during the run, not this
            # method.  Remove any stale copy before launching so that a
            # run which dies before the plugin fires leaves a missing
            # file rather than the previous step's forces to be
            # silently reread.
            for stale in ("MM_FORCES", "QM_NET_FORCE"):
                stale_path = os.path.join(self.directory, stale)
                if os.path.isfile(stale_path):
                    os.remove(stale_path)
        # Reuse the previous step's orbitals and density once they exist.
        if self.frame[0] and os.path.isfile(
                os.path.join(self.directory, "WAVECAR"),
        ):
            tags.setdefault("ISTART", 1)
        vasp_utils.write_incar(
            os.path.join(self.directory, "INCAR"),
            tags,
        )
        return order

    def _write_mm_charges(self) -> int:
        r"""Write subsystem II point charges for the plugin.

        Returns:
            The number of MM charges written.
        """
        os.makedirs(self.directory, exist_ok=True)
        indices = sorted(self.system.select("subsystem II"))
        positions = np.asarray(self.system.positions)[indices]
        charges = np.asarray(self.system.charges)[indices]
        path = os.path.join(self.directory, "MM_CHARGES")
        if os.path.isfile(path):
            os.remove(path)
        vasp_utils.write_mm_charges(
            path, positions, charges, self.frame[0], self.embedding_sigma,
        )
        return len(charges)

    def _write_pme_data(self) -> int:
        r"""Write the PME sources and Ewald parameters for the plugin.

        FFT embedding uses only subsystem III as periodic PME sources.
        The Gaussian FFT already supplies subsystem II and its images;
        VASP treats subsystem I periodically.  Remove I and II from the
        source charges rather than subtracting a single-image erf field.
        The physical charges and MM_CHARGES remain unchanged.

        The experimental erfc route retains its reciprocal II field,
        which is needed to complement the real-space erfc correction.

        Returns:
            The number of charges written.
        """
        os.makedirs(self.directory, exist_ok=True)
        potential = self.potentials[0]
        from .vasp_plugin import erfc_near_enabled
        from pydft_qmmm.embedding.pme_grid import write_pme_data
        charges = np.array(self.system.charges, copy=True)
        if erfc_near_enabled():
            excluded = sorted(self.system.select("subsystem I"))
        else:
            charges[sorted(self.system.select("not subsystem III"))] = 0.
            excluded = []
        path = os.path.join(self.directory, "PME_DATA")
        if os.path.isfile(path):
            os.remove(path)
        write_pme_data(
            path,
            np.asarray(self.system.positions),
            charges,
            excluded,
            potential.pme_alpha,
            tuple(potential.pme_gridnumber),
            potential.pme_spline_order,
            self.frame[0],
        )
        return len(self.system.charges)

    def _plugin_env(self) -> dict[str, str]:
        """Environment for the VASP subprocess."""
        env = dict(os.environ)
        here = os.path.dirname(os.path.abspath(__file__))
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{here}:{existing}" if existing else here
        return env

    def _read_mm_forces(self) -> NDArray[np.float64]:
        r"""Read the QM->MM forces the plugin wrote.

        Returns:
            An Nx3 array of forces
            (:math:`\mathrm{kJ\;mol^{-1}\;\mathring{A}^{-1}}`) ordered to
            match ``sorted(system.select("subsystem II"))``.

        Raises:
            VaspExecutionError: If the file is absent, disagrees with
                subsystem II on the atom count, or holds forces that are
                identically zero -- each of which means the
                back-reaction was not applied.
            ValueError: If the file belongs to a different step.
        """
        path = os.path.join(self.directory, "MM_FORCES")
        if not os.path.isfile(path):
            raise vasp_utils.VaspExecutionError(
                self.directory,
                "MM_FORCES is absent, so the QM->MM back-reaction was "
                "never computed and momentum will not be conserved.",
            )
        # _run increments frame[0] only after the plugin has already
        # stamped MM_FORCES, so the file carries the PREVIOUS counter
        # value.
        forces, _ = vasp_utils.read_mm_forces(
            path, expect_step=self.frame[0] - 1,
        )
        indices = sorted(self.system.select("subsystem II"))
        if len(forces) != len(indices):
            raise vasp_utils.VaspExecutionError(
                self.directory,
                f"MM_FORCES holds {len(forces)} rows but subsystem II "
                f"has {len(indices)} atoms.",
            )
        if len(forces) and not np.any(forces):
            raise vasp_utils.VaspExecutionError(
                self.directory,
                "MM_FORCES is identically zero, which is what the "
                "unimplemented back-reaction looked like.",
            )
        return forces

    def _read_qm_net_force(self) -> NDArray[np.float64]:
        r"""Read the net force VASP removed from the QM ions.

        VASP subtracts the mean force from every ion, which is correct
        for an isolated periodic cell -- the total energy really is
        translationally invariant there, so the net force must vanish --
        but wrong under embedding, where ``V_ext`` breaks that
        invariance and the net force is exactly the momentum the MM
        subsystem transfers to the QM one.  Restoring it is what makes
        Newton's third law hold across the two subsystems.

        Returns:
            The net force
            (:math:`\mathrm{kJ\;mol^{-1}\;\mathring{A}^{-1}}`) on the QM
            subsystem, which VASP's reported forces do not contain.

        Raises:
            VaspExecutionError: If the file is absent.
        """
        path = os.path.join(self.directory, "QM_NET_FORCE")
        if not os.path.isfile(path):
            raise vasp_utils.VaspExecutionError(
                self.directory,
                "QM_NET_FORCE is absent, so the net force VASP removed "
                "from the QM ions cannot be restored and momentum will "
                "not be conserved.",
            )
        with open(path) as fh:
            fh.readline()
            net = np.array(
                [float(value) for value in fh.readline().split()],
            )
        # The plugin writes eV/Angstrom, VASP's own unit.
        return net * KJMOL_PER_EV

    def _check_plugin_fired(self) -> None:
        """Verify that the plugin actually ran.

        Raises:
            VaspExecutionError: If the sentinel is absent, which means
                the external potential was never applied and the energy
                is quietly the unembedded one.
        """
        from .vasp_plugin import SENTINEL
        if not os.path.isfile(os.path.join(self.directory, SENTINEL)):
            raise vasp_utils.VaspExecutionError(
                self.directory,
                "Electrostatic embedding was requested but the plugin "
                "sentinel is absent, so the external potential was "
                "never applied.  Check that vasp_std is built with "
                "-DPLUGINS and that PYTHONHOME and PATH point at the "
                "environment holding the plugin's Python.",
            )

    @system_cache("positions", "elements", "subsystems", "box", "charges")
    def _run(self) -> tuple[float, NDArray[np.float64]]:
        r"""Run VASP on the QM subsystem.

        Returns:
            The energy (:math:`\mathrm{kJ\;mol^{-1}}`) and an Nx3 force
            array (:math:`\mathrm{kJ\;mol^{-1}\;\mathring{A}^{-1}}`) for
            the QM atoms only, ordered to match
            ``sorted(system.select("subsystem I"))``.
        """
        order = self._write_input()
        if self.embedding:
            self._write_mm_charges()
            if self.potentials:
                self._write_pme_data()
        vasp_utils.run_vasp(
            self.command, self.directory,
            env=self._plugin_env() if self.embedding else None,
        )
        if self.embedding:
            self._check_plugin_fired()
        energy, forces = vasp_utils.read_vasprun(
            os.path.join(self.directory, "vasprun.xml"),
        )
        if len(forces) != len(order):
            raise vasp_utils.VaspExecutionError(
                self.directory,
                f"VASP returned forces for {len(forces)} atoms, but "
                f"{len(order)} QM atoms were submitted.",
            )
        # Undo the species grouping applied when writing the POSCAR.
        qm_forces = np.empty_like(forces)
        qm_forces[order, :] = forces
        self.frame[0] += 1
        return energy * KJMOL_PER_EV, qm_forces * KJMOL_PER_EV


class VaspPotential(VaspInterface, AtomicPotential):
    """A potential wrapping VASP functionality.

    Args:
        system: The system that will inform the interface to VASP.
        charge: The net charge (:math:`e`) of the QM subsystem.
        directory: The working directory for VASP calculations.
        command: The shell command that launches VASP.
        incar: INCAR tags applied to every calculation.
        kpts: The number of k-points along each reciprocal lattice
            vector.
        pp_path: The directory containing per-species POTCAR
            subdirectories.
        potcar_map: A mapping from element symbol to POTCAR
            subdirectory name.
    """

    def compute_energy(self) -> float:
        r"""Compute the energy of the system using VASP.

        Returns:
            The energy (:math:`\mathrm{kJ\;mol^{-1}}`) of the QM
            subsystem.
        """
        energy, _ = self._run()
        return energy

    def compute_forces(self) -> NDArray[np.float64]:
        r"""Compute the forces on the system using VASP.

        Returns:
            The forces (:math:`\mathrm{kJ\;mol^{-1}\;\mathring{A}^{-1}}`)
            on every atom in the system, with non-QM atoms zeroed.
        """
        _, qm_forces = self._run()
        forces = np.zeros(self.system.positions.shape)
        qm_indices = sorted(self.system.select("subsystem I"))
        forces[qm_indices, :] = qm_forces
        if self.embedding:
            # Restore the net force VASP removed.
            forces[qm_indices, :] += (
                self._read_qm_net_force() / len(qm_indices)
            )
            # Subsystem III is deliberately left at zero: under direct
            # QM/MM/PME its force from the QM region is the "Y = MM"
            # term, which OpenMM supplies from the static forcefield
            # charges.  See John et al., JCP 161, 034103 (2024), Table I.
            embed_indices = sorted(self.system.select("subsystem II"))
            forces[embed_indices, :] = self._read_mm_forces()
        return forces

    def compute_components(self) -> dict[str, float]:
        r"""Compute the components of energy using VASP.

        Returns:
            An empty dict; the VASP interface does not currently expose
            energy sub-components.
        """
        components: dict[str, float] = {}
        return components
