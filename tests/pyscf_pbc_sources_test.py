"""Periodic source selection and discrete embedding force regressions."""
from types import SimpleNamespace

import numpy as np
import pytest

from pydft_qmmm import Atom, System
from pydft_qmmm.interfaces.pyscf_pbc import pbc_forces
from pydft_qmmm.interfaces.pyscf_pbc.pbc_embedding import external_potential
from pydft_qmmm.interfaces.pyscf_pbc.pbc_interface import _nuclear_coupling
from pydft_qmmm.interfaces.pyscf_pbc.pbc_pme import PeriodicPMEElectronicPotential
from pydft_qmmm.interfaces.vasp.grid_potential import spectral_value_and_gradient
from pydft_qmmm.utils import BOHR_PER_ANGSTROM, KJMOL_PER_EH, Subsystem


@pytest.fixture
def environment():
    system = System([
        Atom(position=np.array([2.3, 2.1, 2.8]), charge=-0.8,
             element="O", subsystem=Subsystem.I),
        Atom(position=np.array([4.2, 3.7, 4.1]), charge=0.4,
             element="H", subsystem=Subsystem.II),
        Atom(position=np.array([7.1, 6.3, 5.4]), charge=0.4,
             element="H", subsystem=Subsystem.III),
    ], box=np.eye(3) * 10.)
    return system, PeriodicPMEElectronicPotential(system, 0.5, (20, 20, 20), 5)


def test_removed_sources_and_partition_changes(environment):
    system, potential = environment
    probes = np.array([[1.2, 3.4, 5.6], [6.7, 4.2, 2.9]])
    charges = np.array(system.charges, copy=True)
    field = potential.compute_potential(probes)
    assert np.max(np.abs(field)) > 1e-6
    np.testing.assert_array_equal(system.charges, charges)
    system.positions[:2] += 0.3
    system.charges[:2] *= 2
    np.testing.assert_array_equal(potential.compute_potential(probes), field)
    reaction = potential.compute_source_forces(probes, np.array([-1., 1.]))
    np.testing.assert_array_equal(reaction[:2], np.zeros((2, 3)))
    system.subsystems[2] = Subsystem.II
    np.testing.assert_array_equal(potential.compute_potential(probes), 0.)
    np.testing.assert_array_equal(
        potential.compute_source_forces(probes, np.array([-1., 1.])), 0.,
    )
    system.subsystems[1] = Subsystem.III
    assert np.max(np.abs(potential.compute_potential(probes))) > 1e-6


def test_interface_adapts_molecular_pme_without_changing_it(environment):
    from pydft_qmmm.interfaces.pyscf_pbc.pbc_factory import pyscf_pbc_interface_factory
    from pydft_qmmm.potentials.pme_potential import PMEElectronicPotential

    system, _ = environment
    molecular = PMEElectronicPotential(system, 0.5, (20, 20, 20), 5)
    interface = pyscf_pbc_interface_factory(
        system, basis="gth-szv", pseudo="gth-pbe", functional="pbe",
        charge=0, multiplicity=1, mesh=(15, 15, 15),
    )
    interface.add_electronic_potential(molecular)
    assert type(molecular) is PMEElectronicPotential
    assert isinstance(interface.potentials[0], PeriodicPMEElectronicPotential)
    assert interface.potentials[0].system is system
    assert interface.potentials[0].pme_alpha == molecular.pme_alpha
    assert interface.potentials[0].pme_gridnumber == molecular.pme_gridnumber
    assert interface.potentials[0].pme_spline_order == molecular.pme_spline_order


@pytest.mark.parametrize("mesh", [(9, 11, 13), (10, 12, 14)])
def test_nuclear_grid_charge_is_interpolation_transpose(mesh):
    box = np.array([[8., 0., 0.], [1., 9., 0.], [0.5, 1.2, 10.]])
    positions = np.array([[2.3, 3.7, 4.2], [4.6, 2.1, 3.8]])
    charges = np.array([6., 1.])
    cell = SimpleNamespace(mesh=mesh, atom_coords=lambda: positions * BOHR_PER_ANGSTROM,
                           natm=2, atom_charge=lambda i: charges[i])
    field = np.random.default_rng(42).normal(size=mesh)
    grid_charge = pbc_forces.nuclear_charge_on_grid(SimpleNamespace(cell=cell), box)
    values, _ = spectral_value_and_gradient(field, box, positions)
    assert grid_charge @ field.ravel() == pytest.approx(charges @ values, abs=1e-12)


@pytest.mark.parametrize("mesh", [(15, 15, 15), (16, 16, 16)])
def test_mm_reaction_differentiates_discrete_energy(environment, monkeypatch, mesh):
    system, potential = environment
    axes = [np.arange(n) * 10. / n for n in mesh]
    coords = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    cell = SimpleNamespace(
        mesh=mesh, vol=(10. * BOHR_PER_ANGSTROM)**3,
        get_uniform_grids=lambda: coords * BOHR_PER_ANGSTROM,
        atom_coords=lambda: np.asarray(system.positions[:1]) * BOHR_PER_ANGSTROM,
        natm=1, atom_charge=lambda i: 6.,
    )
    state = SimpleNamespace(cell=cell, coords=coords, embed_indices=(1,),
                            qm_indices=(0,), potentials=(potential,))
    charge = -np.exp(-np.sum((coords - 2.5)**2, axis=1))
    charge *= 6. / -charge.sum()
    monkeypatch.setattr(pbc_forces, "qm_density_on_grid", lambda *args: charge.copy())

    def energy():
        field = external_potential(system, cell, [potential], [1], 0.5)
        return (-charge @ field + _nuclear_coupling(cell, field, system.box)) * KJMOL_PER_EH

    analytic = pbc_forces.mm_forces(None, state, system, 0.5, len(system))
    analytic *= KJMOL_PER_EH * BOHR_PER_ANGSTROM
    step = 1e-4
    for atom in (1, 2):
        for axis in range(3):
            original = float(system.positions[atom, axis])
            try:
                system.positions[atom, axis] = original + step
                plus = energy()
                system.positions[atom, axis] = original - step
                minus = energy()
            finally:
                system.positions[atom, axis] = original
            assert analytic[atom, axis] == pytest.approx(
                -(plus - minus) / (2 * step), abs=2e-5, rel=2e-6,
            )
