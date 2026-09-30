"""Electrostatic embedding for PySCF."""
from __future__ import annotations

__all__ = [
    "add_finite_embedding",
    "finite_embedding_forces",
    "build_quadrature_grid",
    "pme_ao_operator",
    "pme_qm_forces",
    "pme_qm_forces_and_density",
]

from typing import Any
from typing import TYPE_CHECKING

import numpy as np
from pyscf.dft import gen_grid
from pyscf.dft import numint

from pydft_qmmm.utils import BOHR_PER_ANGSTROM

from .pyscf_backend import load_submodule
from .pyscf_backend import to_numpy
from .pyscf_utils import spin_sum

#: Quadrature points per AO batch.
GRID_BLOCK_SIZE = 16384

if TYPE_CHECKING:
    from collections.abc import Sequence
    from numpy.typing import NDArray
    from pydft_qmmm import System


def add_finite_embedding(
        backend: Any,
        method: Any,
        system: System,
        embed_indices: Sequence[int],
) -> Any:
    r"""Decorate a solver with the finite point-charge environment.

    Args:
        backend: The package providing the solvers, whose own QM/MM
            module has to be used: GPU4PySCF's solvers do not inherit
            from PySCF's, so PySCF's decorator rejects them.
        method: The molecular PySCF solver to decorate.
        system: The system holding the environment positions
            (:math:`\mathrm{\mathring{A}}`) and charges (:math:`e`).
        embed_indices: The original system indices of the point charges
            to embed.

    Returns:
        The decorated solver, or the original solver when there is
        nothing to embed.
    """
    indices = list(embed_indices)
    if not indices:
        return method
    qmmm = load_submodule(backend, "qmmm")
    return qmmm.add_mm_charges(
        method,
        np.asarray(system.positions)[indices],
        np.asarray(system.charges)[indices],
        unit="Angstrom",
    )


def finite_embedding_forces(
        method: Any,
        dm: NDArray[np.float64],
        qm_indices: Sequence[int],
        embed_indices: Sequence[int],
        natoms: int,
) -> NDArray[np.float64]:
    r"""Assemble the QM and MM gradients into full-system forces.

    Args:
        method: The converged PySCF solver.
        dm: The converged density matrix.
        qm_indices: The original system indices of subsystem I, in the
            order they appear in the PySCF molecule.
        embed_indices: The original system indices of the embedded
            point charges, in the order they were embedded.
        natoms: The number of atoms in the full system.

    Returns:
        The forces (:math:`\mathrm{E_h\;a_0^{-1}}`) acting on atoms in
        the system.
    """
    forces = np.zeros((natoms, 3))
    gradient = method.nuc_grad_method()
    gradient.verbose = 0
    forces[list(qm_indices)] -= to_numpy(gradient.kernel())
    if len(embed_indices):
        forces[list(embed_indices)] -= to_numpy(
            gradient.grad_hcore_mm(spin_sum(dm)),
        ) + to_numpy(gradient.grad_nuc_mm())
    return forces


def build_quadrature_grid(
        method: Any,
        grid_level: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Get the molecular quadrature grid to integrate the embedding on.

    A uniform mesh at PME resolution cannot integrate an all-electron
    density: its nuclear cusps are unresolved at half an Angstrom, and
    the midpoint rule over-counts them by more than an order of
    magnitude.  An atom-centered Becke grid puts its radial shells
    inside the cusps instead, integrating the same density to about a
    part in a million with a few tens of thousands of points.

    A Kohn-Sham solver already carries such a grid for its
    exchange-correlation term, and reusing it means the two integrals
    share one quadrature.  A Hartree-Fock solver has none, so one is
    built at the same level.

    Args:
        method: The PySCF solver whose grid will be used.  It is built
            if it has not been already.
        grid_level: The quadrature level to use when the solver carries
            no grid of its own.

    Returns:
        The quadrature coordinates (:math:`\mathrm{\mathring{A}}`) and
        their weights (:math:`\mathrm{a_0^3}`).
    """
    grids = getattr(method, "grids", None)
    if grids is None:
        grids = gen_grid.Grids(method.mol)
        grids.level = grid_level
    if grids.coords is None:
        grids.build()
    coordinates = np.asarray(to_numpy(grids.coords)) / BOHR_PER_ANGSTROM
    return coordinates, np.asarray(to_numpy(grids.weights))


def pme_ao_operator(
        mol: Any,
        potential: NDArray[np.float64],
        coords: NDArray[np.float64],
        weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Contract a sampled potential into a one-electron operator.

    Args:
        mol: The molecular PySCF object for the QM subsystem.
        potential: The potential energy of one electron
            (:math:`\mathrm{E_h}`) at each quadrature point.
        coords: The quadrature coordinates
            (:math:`\mathrm{\mathring{A}}`).
        weights: The quadrature weights (:math:`\mathrm{a_0^3}`).

    Returns:
        The symmetric one-electron operator (:math:`\mathrm{E_h}`) in
        the AO basis.
    """
    coords = np.asarray(coords, dtype=float)
    weighted = (
        np.asarray(potential, dtype=float).reshape(-1)
        * np.asarray(weights, dtype=float).reshape(-1)
    )
    matrix = np.zeros((mol.nao, mol.nao))
    for start in range(0, len(coords), GRID_BLOCK_SIZE):
        block = slice(start, start + GRID_BLOCK_SIZE)
        ao = numint.eval_ao(
            mol, coords[block] * BOHR_PER_ANGSTROM, deriv=0,
        )
        matrix += np.einsum("pi,p,pj->ij", ao, weighted[block], ao)
    # Remove round-off asymmetry before SCF.
    return 0.5 * (matrix + matrix.T)


def pme_qm_forces_and_density(
        mol: Any,
        dm: NDArray[np.float64],
        potential: NDArray[np.float64],
        coords: NDArray[np.float64],
        weights: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Differentiate the embedding operator over the QM atoms.

    Only the AO-center derivative is included.  Grid-response terms are
    quadrature artifacts and vanish as the grid converges.

    For an AO centered on atom :math:`A`,
    :math:`\partial\phi_i/\partial\mathbf{R}_A = -\nabla\phi_i`, and
    both the bra and the ket contribute, giving

    .. math::
        \frac{\partial E}{\partial \mathbf{R}_A}
        = -2 \sum_{i \in A} \sum_j D_{ij}
          \int \nabla\phi_i\, v\, \phi_j.

    Density response enters through PySCF's native gradient.

    Args:
        mol: The molecular PySCF object for the QM subsystem.
        dm: The converged density matrix.
        potential: The potential energy of one electron
            (:math:`\mathrm{E_h}`) at each quadrature point.
        coords: The quadrature coordinates
            (:math:`\mathrm{\mathring{A}}`).
        weights: The quadrature weights (:math:`\mathrm{a_0^3}`).

    Returns:
        The forces (:math:`\mathrm{E_h\;a_0^{-1}}`) on the QM atoms, in
        PySCF atom order, and the electron number density
        (:math:`\mathrm{a_0^{-3}}`) at each quadrature point.
    """
    coords = np.asarray(coords, dtype=float)
    weighted = (
        np.asarray(potential, dtype=float).reshape(-1)
        * np.asarray(weights, dtype=float).reshape(-1)
    )
    total = np.asarray(to_numpy(spin_sum(dm)))
    forces = np.zeros((mol.natm, 3))
    density = np.empty(len(coords))
    slices = mol.aoslice_by_atom()
    for start in range(0, len(coords), GRID_BLOCK_SIZE):
        block = slice(start, start + GRID_BLOCK_SIZE)
        ao = numint.eval_ao(
            mol, coords[block] * BOHR_PER_ANGSTROM, deriv=1,
        )
        # contracted[p, i] = sum_j D_ij phi_j(r_p)
        contracted = ao[0] @ total
        density[block] = np.einsum("pi,pi->p", ao[0], contracted)
        contracted *= weighted[block][:, None]
        derivative = np.einsum("xpi,pi->xi", ao[1:4], contracted)
        for atom, (_, _, first, last) in enumerate(slices):
            forces[atom] += 2.0 * derivative[:, first:last].sum(axis=1)
    return forces, density


def pme_qm_forces(
        mol: Any,
        dm: NDArray[np.float64],
        potential: NDArray[np.float64],
        coords: NDArray[np.float64],
        weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Differentiate the embedding operator with respect to the QM atoms.

    Args:
        mol: The molecular PySCF object for the QM subsystem.
        dm: The converged density matrix.
        potential: The potential energy of one electron
            (:math:`\mathrm{E_h}`) at each quadrature point.
        coords: The quadrature coordinates
            (:math:`\mathrm{\mathring{A}}`).
        weights: The quadrature weights (:math:`\mathrm{a_0^3}`).

    Returns:
        The forces (:math:`\mathrm{E_h\;a_0^{-1}}`) on the QM atoms, in
        PySCF atom order.
    """
    return pme_qm_forces_and_density(
        mol, dm, potential, coords, weights,
    )[0]
