"""The QM/MM/PME-hacked Psi4 software interface.
"""
from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
import psi4.core
from numpy.typing import NDArray

from pydft_qmmm.common import BOHR_PER_ANGSTROM
from pydft_qmmm.common import KJMOL_PER_EH
from pydft_qmmm.interfaces.psi4.psi4_interface import Psi4Interface

if TYPE_CHECKING:
    from pydft_qmmm.interfaces import QMSettings
    from pydft_qmmm.interfaces.psi4.psi4_utils import Psi4Context
    from .psi4_utils import PMEPsi4Options

# psi4.core.be_quiet()


class PMEPsi4Interface(Psi4Interface):
    """A software interface wrapping Psi4 functionality.

    Args:
        settings: The settings used to build the Psi4 interface.
        options: The Psi4 global options derived from the settings.
        functional: The name of the functional to use for
            exchange-correlation calculations.
        context: An object which holds system information to feed into
            Psi4.
    """

    def __init__(
            self,
            settings: QMSettings,
            options: PMEPsi4Options,
            functional: str,
            context: Psi4Context,
    ) -> None:
        self._settings = settings
        self._options = options
        self._functional = functional
        self._context = context
        self._quad_extd_pot = None
        self._nuc_extd_pot = None
        self._nuc_extd_grad = None

    def compute_forces(self) -> NDArray[np.float64]:
        r"""Compute the forces on the system using Psi4.

        Returns:
            The forces (:math:`\mathrm{kJ\;mol^{-1}\;\mathring{A}^{-1}}`) acting
            on atoms in the system.
        """
        wfn = self._generate_wavefunction()
        psi4.set_options(asdict(self._options))
        kwargs: dict[str, NDArray[np.float64]] = {}
        if self._quad_extd_pot is not None:
            kwargs["quad_extd_pot"] = self._quad_extd_pot
        if self._nuc_extd_pot is not None:
            kwargs["nuc_extd_pot"] = self._nuc_extd_pot
        if self._nuc_extd_grad is not None:
            kwargs["nuc_extd_grad"] = self._nuc_extd_grad
        forces = psi4.gradient(
            self._functional,
            ref_wfn=wfn,
            **kwargs,
        )
        forces = forces.to_array() * -KJMOL_PER_EH * BOHR_PER_ANGSTROM
        forces_temp = np.zeros(self._context.positions.shape)
        qm_indices = sorted(self._context.atoms)
        forces_temp[qm_indices, :] = forces[0:len(qm_indices)]
        if self._context.generate_external_potential():
            embed_indices = sorted(self._context.embedding)
            forces = (
                wfn.external_pot().gradient_on_charges().to_array()
                * -KJMOL_PER_EH * BOHR_PER_ANGSTROM
            )
            forces_temp[embed_indices, :] = forces
        forces = forces_temp
        return forces

    @lru_cache
    def _generate_wavefunction(self) -> psi4.core.Wavefunction:
        """Generate the Psi4 Wavefunction object for use by Psi4.

        Returns:
            The Psi4 Wavefunction object, which contains the energy
            and coefficients determined through SCF.
        """
        molecule = self._context.generate_molecule()
        psi4.set_options(asdict(self._options))
        kwargs: dict[str, NDArray[np.float64]] = {}
        if self._quad_extd_pot is not None:
            kwargs["quad_extd_pot"] = self._quad_extd_pot
        if self._nuc_extd_pot is not None:
            kwargs["nuc_extd_pot"] = self._nuc_extd_pot
        if self._nuc_extd_grad is not None:
            kwargs["nuc_extd_grad"] = self._nuc_extd_grad
        _, wfn = psi4.energy(
            self._functional,
            return_wfn=True,
            molecule=molecule,
            external_potentials=self._context.generate_external_potential(),
            **kwargs,
        )
        wfn.to_file(
            wfn.get_scratch_filename(180),
        )
        return wfn

    def compute_quadrature(self) -> NDArray[np.float64]:
        """Build a reference quadrature.

        Returns:
            A reference quadrature constructed from the Psi4
            Geometry object.
        """
        molecule = self._context.generate_molecule()
        sup_func = psi4.driver.dft.build_superfunctional(
            self._functional,
            True,
        )[0]
        c1_molecule = molecule.clone()
        c1_molecule._initial_cartesian = molecule._initial_cartesian.clone()
        c1_molecule.set_geometry(c1_molecule._initial_cartesian)
        c1_molecule.reset_point_group("c1")
        c1_molecule.fix_orientation(True)
        c1_molecule.fix_com(True)
        c1_molecule.update_geometry()
        basis = psi4.core.BasisSet.build(
            c1_molecule,
            "ORBITAL",
        )
        vbase = psi4.core.VBase.build(basis, sup_func, "RV")
        vbase.initialize()
        quadrature = np.concatenate(
            tuple([
                coord.reshape(-1, 1)
                for coord in vbase.get_np_xyzw()[0:3]
            ]),
            axis=1,
        )
        return quadrature

    def update_quad_extd_pot(self, quad_extd_pot: NDArray[np.float64]) -> None:
        r"""Update the external potential at the quadrature grid points.

        Args:
            quad_ext_pot: A potential (:math:`\mathrm{a.u.}\;e^{-1}`)
                evaluated at the quadrature grid points.
        """
        self._quad_extd_pot = quad_extd_pot
        self._generate_wavefunction.cache_clear()

    def update_nuc_extd_pot(self, nuc_extd_pot: NDArray[np.float64]) -> None:
        r"""Update the external potential at the nuclear coordinates.

        Args:
            nuc_ext_pot: A potential (:math:`\mathrm{a.u.}\;e^{-1}`)
                evaluated at the nuclear coordinates.
        """
        self._nuc_extd_pot = nuc_extd_pot
        self._generate_wavefunction.cache_clear()

    def update_nuc_extd_grad(self, nuc_extd_grad: NDArray[np.float64]) -> None:
        r"""Update the gradient of the external potential at the nuclei.

        Args:
            nuc_ext_pot: A gradient of the potential
                (:math:`\mathrm{a.u.}\;e^{-1}\;\mathrm{Bohr^{-1}}`)
                evaluated at the nuclear coordinates.
        """
        self._nuc_extd_grad = nuc_extd_grad
        self._generate_wavefunction.cache_clear()
