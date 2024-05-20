#! /usr/bin/env python3
"""A module for defining the :class:`Simulation` class.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .logger import NullLogger
from pydft_qmmm.calculators import Calculator
from pydft_qmmm.common import Subsystem
from pydft_qmmm.hamiltonians import CalculatorHamiltonian
from pydft_qmmm.hamiltonians import CompositeHamiltonian
from pydft_qmmm.hamiltonians import QMMMHamiltonian

if TYPE_CHECKING:
    from typing import Any
    from pydft_qmmm.hamiltonians import Hamiltonian
    from pydft_qmmm.integrators import Integrator
    from pydft_qmmm.plugins.plugin import Plugin
    from .system import System


class Simulation:
    """An object which manages and performs simulations.

    :param system: |system| to perform calculations on.
    :param hamiltonian: |hamiltonian| to perform calculations with.
    :param integrator: |integrator| to perform calculations with.
    :param logger: |logger| to record data generated during the
        simulation
    :param num_threads: The number of threads to run calculations on.
    :param memory: The amount of memory to allocate to calculations.
    :param plugins: Any :class:`Plugin` objects to apply to the
        simulation.
    """

    def __init__(
            self,
            system: System,
            integrator: Integrator,
            hamiltonian: Hamiltonian | None = None,
            calculator: Calculator | None = None,
            plugins: list[Plugin] | None = None,
            frame: int = 0,
            logger: Any = NullLogger,
    ) -> None:
        # Make initial assignments.
        self.system = system
        self.integrator = integrator
        self.hamiltonian = hamiltonian
        if isinstance(calculator, Calculator):
            self.calculator = calculator
        elif (
            isinstance(hamiltonian, CompositeHamiltonian)
            or isinstance(hamiltonian, CalculatorHamiltonian)
        ):
            self.calculator = hamiltonian.build_calculator(self.system)
        else:
            raise TypeError
        if plugins is None:
            plugins = []
        self.plugins = plugins
        self.frame = frame
        self.logger = logger
        # Perform additional simulation setup
        self._register_plugins()
        self._offset = np.zeros(self.system.positions.shape)
        self._procedure = []
        if Subsystem.I in self.system.subsystems:
            self.center_positions()
            self._procedure.append(self.center_positions)
        if isinstance(self.hamiltonian, CompositeHamiltonian):
            for hamiltonian in self.hamiltonian.hamiltonians:
                if isinstance(hamiltonian, QMMMHamiltonian):
                    self.embedding_cutoff = hamiltonian.embedding_cutoff
                    self.generate_embedding()
                    self._procedure.append(self.generate_embedding)
        self.wrap_positions()
        self._procedure.append(self.wrap_positions)
        self.calculate_energy_forces()

    def run_dynamics(self, steps: int) -> None:
        """Run simulation using the :class:`System`,
        :class:`Calculator`, and :class:`Integrator`.

        :param steps: The number of steps to take.
        """
        with self.logger as logger:
            logger.record(self)
            for i in range(steps):
                new_positions, new_velocities = self.integrator.integrate(
                    self.system,
                )
                self.system.positions = new_positions
                self.system.velocities = new_velocities
                for procedure in self._procedure:
                    procedure()
                self.calculate_energy_forces()
                self.frame += 1
                logger.record(self)

    def calculate_energy_forces(self) -> None:
        """Update the :class:`System` forces and :class:`Simulation`
        energy using calculations from the :class:`Calculator`.
        """
        results = self.calculator.calculate()
        self.system.forces = results.forces
        kinetic_energy = self.integrator.compute_kinetic_energy(
            self.system,
        )
        energy = {
            "Total Energy": kinetic_energy + results.energy,
            ".": {
                "Kinetic Energy": kinetic_energy,
                "Potential Energy": results.energy,
                ".": results.components,
            },
        }
        self.energy = energy

    def center_positions(self) -> None:
        """
        """
        atoms = self.system.subsystem_map[Subsystem.I]
        box = self.system.box
        center = 0.5*box.sum(axis=0)
        positions = self.system.positions
        centroid = np.average(positions[list(atoms), :], axis=0)
        differential = center - centroid
        new_positions = positions + differential
        self._offset += positions - new_positions
        self.system.positions = new_positions

    def wrap_positions(self) -> None:
        """Atoms are wrapped to stay inside of the periodic box.  This
        function ensures molecules are not broken up by a periodic
        boundary, since OpenMM electrostatics will be incorrect if atoms
        in a molecule are not on the same side of the periodic box.
        This method currently assumes an isotropic box.
        """
        box = self.system.box
        inverse_box = np.linalg.inv(box)
        positions = self.system.positions
        new_positions = np.zeros(positions.shape)
        for atoms in self.system.molecule_map.values():
            atoms = list(atoms)
            molecule_positions = positions[atoms, :]
            molecule_centroid = np.average(molecule_positions, axis=0)
            inverse_centroid = molecule_centroid @ inverse_box
            mask = np.floor(inverse_centroid)
            diff = (-mask @ box).reshape((-1, 3))
            temp = molecule_positions + diff[:, np.newaxis, :]
            new_positions[atoms] = temp.reshape((len(atoms), 3))
        self._offset += positions - new_positions
        self.system.positions = new_positions

    def generate_embedding(self) -> None:
        """Create the embedding list for the current :class:`System`,
        as well as the corrective Coulomb potential and forces.

        The distances from the QM atoms are computed using the centroid
        of the non-QM molecule from the centroid of the QM atoms.

        :return: The corrective Coulomb energy and forces for the
            embedded point charges, and the charge field for the QM
            calculation.
        """
        qm_region = list(self.system.qm_region)
        qm_centroid = np.average(
            self.system.positions[qm_region, :],
            axis=0,
        )
        embedding = []
        for atoms in self.system.molecule_map.values():
            atoms = list(atoms)
            nth_centroid = np.average(
                self.system.positions[atoms, :],
                axis=0,
            )
            r_vector = nth_centroid - qm_centroid
            distance = np.sum(r_vector**2)**0.5
            not_qm = set(qm_region).isdisjoint(set(atoms))
            if distance < self.embedding_cutoff and not_qm:
                embedding.extend(atoms)
        # Update the topology with the current embedding atoms.
        temp = [Subsystem.III]*len(self.system)
        temp = [
            Subsystem.II if i in embedding else x
            for i, x in enumerate(temp)
        ]
        temp = [
            Subsystem.I if i in qm_region else x
            for i, x in enumerate(temp)
        ]
        self.system.subsystems = temp

    def _legacy_embedding(self) -> None:
        """Create the embedding list for the current :class:`System`,
        as well as the corrective Coulomb potential and forces.

        The distances from the QM atoms are computed using the centroid
        of the non-QM molecule from the centroid of the QM atoms.

        :return: The corrective Coulomb energy and forces for the
            embedded point charges, and the charge field for the QM
            calculation.
        """
        qm_region = list(self.system.qm_region)
        qm_centroid = np.average(
            self.system.positions[qm_region, :],
            axis=0,
        )
        embedding = []
        for atoms in self.system.molecule_map.values():
            atoms = list(atoms)
            nth_centroid = self.system.positions[atoms[0], :]*1
            r_vector = nth_centroid - qm_centroid
            distance = np.sum(r_vector**2)**0.5
            not_qm = set(qm_region).isdisjoint(set(atoms))
            if distance < self.embedding_cutoff and not_qm:
                embedding.extend(atoms)
        # Update the topology with the current embedding atoms.
        temp = [Subsystem.III]*len(self.system)
        temp = [
            Subsystem.II if i in embedding else x
            for i, x in enumerate(temp)
        ]
        temp = [
            Subsystem.I if i in qm_region else x
            for i, x in enumerate(temp)
        ]
        self.system.subsystems = temp

    def _register_plugins(self) -> None:
        """Dynamically load :class:`Plugin` objects.
        """
        for plugin in self.plugins:
            getattr(self, plugin._key).register_plugin(plugin)
