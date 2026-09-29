"""The Psi4 software interface and potential.

This module contains the software interface for storing and manipulating
Psi4 data types and the potential using Psi4 to calculate energies
and forces.
"""
from __future__ import annotations

import textwrap
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

import numpy as np
import psi4

from pydft_qmmm.interfaces import QMInterface
from pydft_qmmm.potentials import AtomicPotential
from pydft_qmmm.utils import BOHR_PER_ANGSTROM
from pydft_qmmm.utils import KJMOL_PER_EH
from pydft_qmmm.utils import PyDFTQMMMException
from pydft_qmmm.utils import system_cache

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray
    from pydft_qmmm.potentials import ElectronicPotential
    from pydft_qmmm import System  # noqa: F401
    from . import psi4_utils


@dataclass(frozen=True)
class Psi4Interface(QMInterface):
    r"""A mix-in for storing and manipulating Psi4 data types.

    Args:
        system: The system that will inform the interface to the
            external software.
        functional: The name of the functional to use in QM
            calculations.
        charge: The net charge (:math:`e`) of the QM subsystem.
        multiplicity: The spin multiplicity of the QM subsystem.
        output_file: The file to which Psi4 output is written.
        output_interval: The interval at which Psi4 output should be
            written, e.g., the default value of 1 means that output
            will be written every calculation.

    Attributes:
        potentials: A list of electronic potentials to incorporate into
            QM calculations.
        frame: The estimated current frame for output writing purposes.
    """
    functional: str
    charge: int
    multiplicity: int
    output_file: str
    output_interval: int
    potentials: list[ElectronicPotential] = field(
        default_factory=list,
        init=False,
    )
    frame: list[int] = field(
        default_factory=lambda: [0],
        init=False,
    )

    def add_electronic_potential(self, potential: ElectronicPotential) -> None:
        """Add an electronic potential to apply before calculations.

        Args:
            potential: The electronic potential to incorporate into
                QM calculations.
        """
        self.potentials.append(potential)

    @system_cache("positions", "charges", "elements", "subsystems")
    def _generate_wavefunction(self) -> psi4.core.Wavefunction:
        """Generate the Psi4 Wavefunction object.

        Returns:
            The Psi4 Wavefunction object, which contains the energy
            and coefficients determined through SCF.
        """
        if not self.frame[0] % self.output_interval:
            psi4.core.set_output_file(self.output_file, True)
        molecule = self._generate_molecule()
        _, wfn = psi4.energy(
            self.functional,
            return_wfn=True,
            molecule=molecule,
            external_potentials=self._generate_external_potentials(),
        )
        wfn.to_file(
            wfn.get_scratch_filename(180),
        )
        if not self.frame[0] % self.output_interval:
            psi4.core.set_output_file("/dev/null", True)
        self.frame[0] += 1
        return wfn

    @system_cache("positions", "elements", "subsystems")
    def _generate_molecule(self) -> psi4.core.Molecule:
        """Generate the Psi4 Molecule object.

        Returns:
            The Psi4 Molecule object, which contains the positions,
            net charge, and net spin of atoms in the QM subsystem.
        """
        geometrystring = """\n"""
        geometrystring += str(self.charge) + " "
        geometrystring += str(self.multiplicity) + "\n"
        geometrystring += "symmetry c1\n"
        geometrystring += "no_reorient\nno_com\n"
        atoms = sorted(self.system.select("subsystem I"))
        for atom in atoms:
            geometrystring = (
                geometrystring
                + str(self.system.elements[atom]) + str(atom) + " "
                + str(self.system.positions[atom][0]) + " "
                + str(self.system.positions[atom][1]) + " "
                + str(self.system.positions[atom][2]) + "\n"
            )
        molecule = psi4.geometry(geometrystring)
        c1_molecule = molecule.clone()
        c1_molecule._initial_cartesian = molecule._initial_cartesian.clone()
        c1_molecule.set_geometry(c1_molecule._initial_cartesian)
        c1_molecule.reset_point_group("c1")
        c1_molecule.fix_orientation(True)
        c1_molecule.fix_com(True)
        c1_molecule.update_geometry()
        return c1_molecule

    @system_cache("positions", "charges", "subsystems")
    def _generate_external_potentials(self) -> list[Any] | None:
        r"""Generate the data structure needed to perform embedding.

        Returns:
            The list of coordinates (:math:`\mathrm{a.u.}`) and charges
            (:math:`e`) that will be electrostatically embedded by Psi4
            during calculations.
        """
        embedding = sorted(self.system.select("subsystem II"))
        external_potentials: list[Any] = []
        point_charges = []
        for i in embedding:
            point_charges.append(
                (
                    self.system.charges[i],
                    [
                        self.system.positions[i, 0] * BOHR_PER_ANGSTROM,
                        self.system.positions[i, 1] * BOHR_PER_ANGSTROM,
                        self.system.positions[i, 2] * BOHR_PER_ANGSTROM,
                    ],
                ),
            )
        if embedding:
            external_potentials.append(point_charges)
        else:
            external_potentials.append(None)
        # PyDFT-QMMM does not currently support diffuse embedding.
        external_potentials.append(None)
        if self.potentials:
            numint = self._generate_numinthelper()
            v_grid = self._generate_potential()
            potential = numint.potential_integral(v_grid).np
            external_potentials.append(potential)
        else:
            external_potentials.append(None)
        if external_potentials is [None, None, None]:
            return None
        return external_potentials

    @system_cache("positions", "elements", "subsystems")
    def _generate_numinthelper(self) -> psi4.core.NumIntHelper:
        """Generate the Psi4 NumIntHelper object.

        Returns:
            The Psi4 NumIntHelper object, which can evaluate integrals
            for arbitrary potentials via numerical quadrature.
        """
        molecule = self._generate_molecule()
        basis_set = psi4.core.BasisSet.build(
            molecule,
            "BASIS",
            psi4.core.get_global_option("BASIS"),
        )
        grid = psi4.core.DFTGrid.build(molecule, basis_set)
        numinthelper = psi4.core.NumIntHelper(grid)
        return numinthelper

    @system_cache("positions", "elements", "subsystems")
    def _generate_potential(self) -> list[psi4.core.Vector]:
        """Generate the potential grid from ElectronicPotential objects.

        Returns:
            A list of Psi4 Vector objects containing the potential
            evaluated at the points on the Psi4 DFTGrid object.
        """
        numint = self._generate_numinthelper()
        blocks = []
        indices = []
        i = 0
        for block in numint.numint_grid().blocks():
            x = block.x().np
            y = block.y().np
            z = block.z().np
            blocks.append(np.stack((x, y, z), axis=-1))
            indices.append(i)
            i += block.npoints()
        xyz = np.concatenate(tuple(blocks), axis=0)
        v = np.zeros_like(xyz[:, 0])
        for potential in self.potentials:
            v += potential.compute_potential(
                xyz / BOHR_PER_ANGSTROM,
            ).flatten()
        v_split = np.split(v, indices)
        v_grid = [psi4.core.Vector.from_array(x) for x in v_split[1:]]
        return v_grid

    def update_options(self, **kwargs: psi4_utils.Psi4Options) -> None:
        """Set additional options for Psi4.

        Args:
            kwargs: Additional options to provide to Psi4.  See
                `Psi4 options`_ for additional Psi4 options.
        """
        psi4.set_options(kwargs)


class Psi4Potential(Psi4Interface, AtomicPotential):
    """A potential wrapping Psi4 functionality.

    Args:
        system: The system that will inform the interface to the
            external software.
        functional: The name of the functional to use in QM
            calculations.
        charge: The net charge (:math:`e`) of the QM subsystem.
        multiplicity: The spin multiplicity of the QM subsystem.
        output_file: The file to which Psi4 output is written.
        output_interval: The interval at which Psi4 output should be
            written, e.g., the default value of 1 means that output
            will be written every calculation.

    Attributes:
        potentials: A list of electronic potentials to incorporate into
            QM calculations.
        frame: The estimated current frame for output writing purposes.
    """

    def compute_energy(self) -> float:
        r"""Compute the energy of the system using Psi4.

        Returns:
            The energy (:math:`\mathrm{kJ\;mol^{-1}}`) of the system.
        """
        wfn = self._generate_wavefunction()
        return wfn.energy() * KJMOL_PER_EH

    def compute_forces(self) -> NDArray[np.float64]:
        r"""Compute the forces on the system using Psi4.

        Returns:
            The forces (:math:`\mathrm{kJ\;mol^{-1}\;\mathring{A}^{-1}}`)
            acting on atoms in the system.
        """
        qm_indices = sorted(self.system.select("subsystem I"))
        forces_temp = np.zeros(self.system.positions.shape)
        wfn = self._generate_wavefunction()
        grads = psi4.gradient(
            self.functional,
            ref_wfn=wfn,
        )
        forces = grads.np * -KJMOL_PER_EH * BOHR_PER_ANGSTROM
        forces_temp[qm_indices, :] += forces
        if self._generate_external_potentials() is None:
            return forces_temp
        if self._generate_external_potentials()[0] is not None:
            embed_indices = sorted(self.system.select("subsystem II"))
            grads = wfn.external_pot().gradient_on_charges()
            # `grads` will be `None` if a numerical gradient is
            # performed in Psi4, and so the following block
            # restores the analytic gradient on from and on the
            # embedded point charges.  This requires at least Psi4 1.11.
            if grads is None:
                if not hasattr(
                        wfn.external_pot(),
                        "computePotentialGradients",
                ):
                    raise PyDFTQMMMException(
                        textwrap.fill(
                            "\nGradients on embedded point charges "
                            "were not calculated, likely because Psi4 "
                            "opted for finite-difference gradients.  "
                            "PyDFT-QMMM can still calculate the "
                            "gradients on embedded point charges, but "
                            "only using Psi4 v1.11 or higher.  The "
                            "current Psi4 installation does not meet "
                            f"this requirement (v{psi4.__version__} is "
                            "currently installed).",
                        ),
                    )
                D = wfn.Da()
                D.add(wfn.Db())
                grads = wfn.external_pot().computePotentialGradients(
                    wfn.basisset(),
                    D,
                )
                forces = grads.np * -KJMOL_PER_EH * BOHR_PER_ANGSTROM
                forces_temp[qm_indices, :] += forces
                grads = wfn.external_pot().gradient_on_charges()
            forces = grads.np * -KJMOL_PER_EH * BOHR_PER_ANGSTROM
            forces_temp[embed_indices, :] += forces
        if self._generate_external_potentials()[2] is not None:
            numint = self._generate_numinthelper()
            v_grid = self._generate_potential()
            D = wfn.Da()
            D.add(wfn.Db())
            grads = numint.potential_gradient(v_grid, D)
            forces = grads.np * -KJMOL_PER_EH * BOHR_PER_ANGSTROM
            forces_temp[qm_indices, :] += forces
        return forces_temp

    def compute_components(self) -> dict[str, float]:
        r"""Compute the components of energy using Psi4.

        Returns:
            The components of the energy (:math:`\mathrm{kJ\;mol^{-1}}`)
            of the system.
        """
        components: dict[str, float] = {}
        return components
