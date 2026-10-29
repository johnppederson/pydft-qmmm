"""The VASP software interface and potential.

This module wraps VASP as a QM potential within PyDFT-QMMM.  Each energy
or force evaluation writes a fresh set of VASP input files into a
persistent working directory.
"""
from __future__ import annotations

__all__ = ["VaspInterface", "VaspPotential"]

import os
import json
from contextlib import contextmanager
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
    pme_grid_on_qm_mesh: bool = field(default=False, init=False)
    persistent_steps: int = 1
    persistent_timeout: float = 3600.
    qm_box: Any = None
    qm_origin: Any = None
    embedding_grid_shape: Any = None
    slab_correction: bool = False
    qm_slab_correction: bool = False
    check_convergence: bool = False
    _session: list = field(default_factory=list, init=False, repr=False)
    _run_steps: list[int] = field(default_factory=list, init=False, repr=False)
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
        return ElectrostaticCouplingMode.PERIODIC

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
        if self.embedding:
            from .vasp_plugin import _validate_embedding_options
            _validate_embedding_options()
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
        metadata = os.path.join(self.directory, "QM_CELL.json")
        if self.qm_box is not None:
            from pydft_qmmm.potentials.embedding.cell_mapping import (
                CellMapping,
            )
            cell = np.asarray(self.qm_box, dtype=float).T
            origin = np.asarray(self.qm_origin, dtype=float)
            shape = tuple(int(self.incar[k]) for k in ("NGXF", "NGYF", "NGZF"))
            mapping = CellMapping(
                np.asarray(self.system.box).T, cell,
                origin, self.embedding_grid_shape, shape,
            )
            # MM sources live on the full MM grid, including outside the QM
            # cell.
            mapping.validate_positions(np.asarray(
                self.system.positions,
            )[qm_indices])
            positions = positions - origin
            with open(metadata, "w") as fh:
                json.dump(mapping.metadata(), fh)
        elif os.path.exists(metadata):
            os.remove(metadata)
        self._write_slab_data()
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
        if self._uses_persistent:
            # PyDFT-QMMM owns the dynamics; VASP waits for each geometry.
            if self.incar.get("PLUGINS/MODE", "serial").lower() != "serial":
                raise ValueError("Persistent VASP requires serial plugins")
            if int(self.incar.get("ISYM", 0)) != 0:
                raise ValueError("Persistent VASP requires ISYM=0")
            if int(self.incar.get("IWAVPR", 11)) not in (0, 11, 12, 13):
                raise ValueError("Persistent VASP requires IWAVPR=0,11,12,13")
            # Disable VASP's ionic stopping test. Electronic defaults
            # remain VASP's responsibility in each mode.
            tags.update({
                "NSW": self._evaluation_count, "IBRION": 12,
                "EDIFFG": 0., "PLUGINS/STRUCTURE": True,
                "IWAVPR": self.incar.get("IWAVPR", 11),
            })
        else:
            tags.update({"NSW": 0, "IBRION": -1})
            tags.pop("PLUGINS/STRUCTURE", None)
            if self.frame[0] and os.path.isfile(
                    os.path.join(self.directory, "WAVECAR"),
            ):
                tags.setdefault("ISTART", 1)
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
            for stale in (
                "MM_FORCES", "PME_FORCES", "QM_NET_FORCE", "SLAB_FORCES",
                "PLUGIN_ENERGY", "PLUGIN_FIRED.txt",
            ):
                stale_path = os.path.join(self.directory, stale)
                if os.path.isfile(stale_path):
                    os.remove(stale_path)
        if self._uses_persistent:
            mode = "a" if self.embedding else "w"
            plugin = os.path.join(self.directory, "vasp_plugin.py")
            with open(plugin, mode) as fh:
                fh.write(
                    "\nfrom pydft_qmmm.interfaces.vasp.vasp_persistent "
                    "import structure\n",
                )
        vasp_utils.write_incar(
            os.path.join(self.directory, "INCAR"),
            tags,
        )
        return order

    def _write_slab_data(self) -> None:
        """Refresh the slab moment and frame stamp before evaluating forces."""
        qm_indices = sorted(self.system.select("subsystem I"))
        cell = np.asarray(
            self.system.box if self.qm_box is None else self.qm_box,
        ).T
        origin = (
            np.zeros(3) if self.qm_origin is None
            else np.asarray(self.qm_origin)
        )
        slab_path = os.path.join(self.directory, "SLAB_DATA.json")
        if self.slab_correction:
            from pydft_qmmm.potentials.embedding.slab import coefficient
            from pydft_qmmm.potentials.embedding.slab import neutral_dipole
            if not self.embedding:
                raise ValueError("Slab correction requires embedded VASP")
            truncated = self.incar.get("KERNEL_TRUNCATION/LTRUNCATE", False)
            if self.qm_slab_correction:
                if truncated:
                    raise ValueError(
                        "QM slab correction requires periodic QM without "
                        "Coulomb truncation",
                    )
                if self.incar.get("LDIPOL", False):
                    raise ValueError(
                        "QM slab correction requires LDIPOL=False to avoid "
                        "double counting",
                    )
            elif self.incar.get("LDIPOL", False):
                # VASP's own dipole correction handles QM-QM; the step must sit
                # at the QM cell face, mid-vacuum.
                dipol = np.asarray(self.incar.get("DIPOL", []), dtype=float)
                if (
                    truncated or self.incar.get("IDIPOL") != 3
                    or dipol.shape != (3,)
                    or not np.isclose(dipol[2] % 1.0, 0.5)
                ):
                    raise ValueError(
                        "LDIPOL slab correction requires IDIPOL=3, "
                        "DIPOL z=0.5, and no Coulomb truncation",
                    )
            elif (
                truncated is not True
                or self.incar.get("KERNEL_TRUNCATION/IDIMENSIONALITY") != 2
                or self.incar.get("KERNEL_TRUNCATION/ISURFACE") != 3
            ):
                raise ValueError(
                    "Slab correction requires z kernel truncation, LDIPOL, "
                    "or qm_slab_correction=True",
                )
            charges = np.array(self.system.charges, copy=True)
            charges[qm_indices] = 0.
            moment = neutral_dipole(self.system.positions, charges)
            with open(slab_path, "w") as fh:
                json.dump(
                    dict(
                        coefficient=coefficient(self.system.box),
                        moment=moment,
                        charges=charges.tolist(), origin=origin.tolist(),
                        qm_self=self.qm_slab_correction,
                        qm_coefficient=coefficient(cell),
                        step=self.frame[0],
                    ), fh,
                )
        elif os.path.exists(slab_path):
            os.remove(slab_path)

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

        Returns:
            The number of charges written.
        """
        from .vasp_plugin import _validate_embedding_options
        _validate_embedding_options()
        os.makedirs(self.directory, exist_ok=True)
        potential = self.potentials[0]
        from pydft_qmmm.potentials.embedding.pme_grid import write_pme_data
        charges = np.array(self.system.charges, copy=True)
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
            ((0, 0, 0) if self.pme_grid_on_qm_mesh
             else tuple(potential.pme_gridnumber)),
            potential.pme_spline_order,
            self.frame[0],
        )
        return len(self.system.charges)

    def _plugin_env(self) -> dict[str, str]:
        """Environment for the VASP subprocess."""
        env = dict(os.environ)
        here = os.path.dirname(os.path.abspath(__file__))
        package_root = os.path.dirname(os.path.dirname(os.path.dirname(here)))
        here = os.pathsep.join(
            (os.path.abspath(self.directory), package_root, here),
        )
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

    def _read_pme_forces(self) -> NDArray[np.float64]:
        r"""Read the reciprocal QM->MM forces the plugin wrote.

        Returns:
            An Nx3 array of forces
            (:math:`\mathrm{kJ\;mol^{-1}\;\mathring{A}^{-1}}`), one row
            per system atom, nonzero only for the PME_DATA sources.

        Raises:
            VaspExecutionError: If the file is absent or does not cover
                every atom.
        """
        path = os.path.join(self.directory, "PME_FORCES")
        if not os.path.isfile(path):
            raise vasp_utils.VaspExecutionError(
                self.directory,
                "PME_FORCES is absent, so the reciprocal QM->MM "
                "back-reaction on subsystem III was never computed.",
            )
        # Stamped like MM_FORCES, with the previous frame counter.
        forces, _ = vasp_utils.read_mm_forces(
            path, expect_step=self.frame[0] - 1,
        )
        if len(forces) != len(self.system):
            raise vasp_utils.VaspExecutionError(
                self.directory,
                f"PME_FORCES holds {len(forces)} rows but the system "
                f"has {len(self.system)} atoms.",
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
        if self._uses_persistent:
            return self._run_persistent()
        return self._run_static()

    def _run_static(self):
        """Evaluate one geometry with VASP's static calculation mode."""
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
        if self.check_convergence:
            self._check_convergence(1)
        energy, forces = vasp_utils.read_vasprun(
            os.path.join(self.directory, "vasprun.xml"),
        )
        if self.embedding:
            energy += vasp_utils.plugin_energy_rounding(self.directory)
        if len(forces) != len(order):
            raise vasp_utils.VaspExecutionError(
                self.directory, "VASP force count does not match QM atoms",
            )
        qm_forces = np.empty_like(forces)
        qm_forces[order, :] = forces
        self.frame[0] += 1
        return energy * KJMOL_PER_EV, qm_forces * KJMOL_PER_EV

    def _persistent_configuration(self):
        """Settings that VASP cannot change while its process is running."""
        qm = sorted(self.system.select("subsystem I"))
        tags = {
            k: v for k, v in self.incar.items()
            if k not in ("ISTART", "ICHARG")
        }
        return (
            tuple(qm), tuple(self.system.elements[qm]),
            np.asarray(self.system.box).tobytes(), repr(sorted(tags.items())),
            self.kpts, self.pp_path, repr(sorted(self.potcar_map.items())),
            self.embedding, self.embedding_sigma, self.slab_correction,
            self.qm_slab_correction,
            None if self.qm_box is None else np.asarray(self.qm_box).tobytes(),
            None if self.qm_origin is None else np.asarray(
                self.qm_origin,
            ).tobytes(),
            None if self.embedding_grid_shape is None else tuple(
                self.embedding_grid_shape,
            ),
        )

    def _run_persistent(self):
        from .vasp_persistent import PersistentVasp
        configuration = self._persistent_configuration()
        if self._session:
            session, initial, order = self._session
            if configuration != initial:
                session.close()
                raise ValueError(
                    "Persistent VASP cell, QM membership, or electronic "
                    "settings changed",
                )
            if session.closed:
                raise ValueError(
                    "Persistent VASP session is closed; create a new "
                    "calculator",
                )
        else:
            order = self._write_input()
            # One log for the whole session; frames are told apart below.
            energy_log = os.path.join(self.directory, "PLUGIN_ENERGY")
            if os.path.isfile(energy_log):
                os.remove(energy_log)
        try:
            # VASP is either not launched yet or blocked inside structure().
            if self.embedding:
                from .vasp_plugin import SENTINEL
                for name in (
                        "MM_FORCES", "PME_FORCES", "QM_NET_FORCE",
                        "SLAB_FORCES", SENTINEL,
                ):
                    path = os.path.join(self.directory, name)
                    if os.path.isfile(path):
                        os.remove(path)
                self._write_mm_charges()
                if self.potentials:
                    self._write_pme_data()
            self._write_slab_data()
            if not self._session:
                session = PersistentVasp(
                    self.command, self.directory, self._plugin_env(),
                    self._evaluation_count, self.persistent_timeout,
                )
                self._session.extend([session, configuration, order])
            indices = sorted(self.system.select("subsystem I"))
            positions = np.asarray(self.system.positions)[indices][order]
            cell = np.asarray(self.system.box).T
            if self.qm_box is not None:
                from pydft_qmmm.potentials.embedding.cell_mapping import (
                    CellMapping,
                )
                cell = np.asarray(self.qm_box).T
                mapping = CellMapping(
                    np.asarray(self.system.box).T, cell, self.qm_origin,
                    self.embedding_grid_shape,
                    tuple(int(self.incar[k])
                          for k in ("NGXF", "NGYF", "NGZF")),
                )
                mapping.validate_positions(positions)
                positions = positions - np.asarray(self.qm_origin)
            fractional = positions @ np.linalg.inv(cell)
            energy, forces = session.evaluate(fractional)
            if (
                    self.check_convergence
                    and session.frame == self._evaluation_count
            ):
                self._check_convergence(self._evaluation_count)
            if self.embedding:
                self._check_plugin_fired()
                # VASP stores the plugin energy in single precision.
                energy += vasp_utils.plugin_energy_rounding(
                    self.directory, frame=session.frame - 1,
                )
        except BaseException:
            if self._session:
                self._session[0].close()
            raise
        qm_forces = np.empty_like(forces)
        qm_forces[order, :] = forces
        self.frame[0] += 1
        return energy * KJMOL_PER_EV, qm_forces * KJMOL_PER_EV

    def _check_convergence(self, expected_steps: int) -> None:
        """Validate completed SCFs after VASP has flushed its output."""
        path = os.path.join(self.directory, "OUTCAR")
        try:
            with open(path) as stream:
                count = sum(
                    'aborting loop because EDIFF is reached' in line
                    for line in stream
                )
        except OSError as error:
            raise vasp_utils.VaspExecutionError(
                self.directory, "Cannot read OUTCAR to check SCF convergence",
            ) from error
        if count != expected_steps:
            raise vasp_utils.VaspExecutionError(
                self.directory,
                f"Only {count}/{expected_steps} SCFs converged",
            )

    @property
    def _uses_persistent(self) -> bool:
        return bool(self._run_steps) or self.persistent_steps > 1

    @property
    def _evaluation_count(self) -> int:
        return self._run_steps[0] if self._run_steps else self.persistent_steps

    @contextmanager
    def calculation_run(self, evaluations: int):
        """Run a complete dynamics segment and release VASP on all exits."""
        if self._run_steps:
            raise RuntimeError("A VASP calculation run is already active")
        if self._session and not self._session[0].closed:
            raise RuntimeError(
                "Close the active VASP run before starting dynamics",
            )
        self._session.clear()
        self.frame[0] = 0
        # Include a fresh starting geometry, even after a previous segment.
        self._run.cache_clear()
        self._run_steps.append(evaluations)
        try:
            yield
        finally:
            try:
                self.close()
            finally:
                self._run_steps.clear()
                # An SP at the same geometry must use static electronic
                # defaults instead of returning the last MD result.
                self._run.cache_clear()

    def close(self):
        """Release the persistent process, including on early termination."""
        if self._session:
            self._session[0].close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


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
            embed_indices = sorted(self.system.select("subsystem II"))
            forces[embed_indices, :] = self._read_mm_forces()
            if self.potentials:
                # PMEExcluded cancels OpenMM's force-field QM charges on
                # subsystem III, so its force from the QM density and
                # nuclei comes from the reciprocal sum here.
                third = sorted(self.system.select("subsystem III"))
                forces[third, :] = self._read_pme_forces()[third]
        if self.slab_correction:
            slab_forces, _ = vasp_utils.read_mm_forces(
                os.path.join(self.directory, "SLAB_FORCES"), self.frame[0] - 1,
            )
            if slab_forces.shape != forces.shape:
                raise ValueError("Incorrect slab reaction force shape")
            forces += slab_forces
        return forces

    def compute_components(self) -> dict[str, float]:
        r"""Compute the components of energy using VASP.

        Returns:
            An empty dict; the VASP interface does not currently expose
            energy sub-components.
        """
        components: dict[str, float] = {}
        return components
