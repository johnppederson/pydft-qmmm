"""Simulation coordinate controls without an external QM engine."""
import numpy as np
import pytest

from pydft_qmmm import Atom, LangevinIntegrator, Simulation, System, VerletIntegrator
from pydft_qmmm.calculators import CompositeCalculator
from pydft_qmmm.plugins import Stationary
from pydft_qmmm.utils import Subsystem


def make_system():
    return System([
        Atom(element='H', mass=1., residue=0, subsystem=Subsystem.I,
             position=np.array([1., 2., 3.])),
        Atom(element='H', mass=1., residue=1, subsystem=Subsystem.II,
             position=np.array([12., -1., 4.])),
    ], box=np.eye(3) * 10)


@pytest.mark.parametrize('options, expected', [
    ({}, [[5., 5., 5.], [6., 2., 6.]]),
    ({'center': True, 'wrap': True}, [[5., 5., 5.], [6., 2., 6.]]),
    ({'center': False, 'wrap': False}, [[1., 2., 3.], [12., -1., 4.]]),
    ({'center': True, 'wrap': False}, [[5., 5., 5.], [16., 2., 6.]]),
    ({'center': False, 'wrap': True}, [[1., 2., 3.], [2., 9., 4.]]),
])
def test_coordinate_options(options, expected, no_logging):
    system = make_system()
    simulation = Simulation(system, VerletIntegrator(1),
                            calculator=CompositeCalculator(system, []),
                            **options, **no_logging)
    simulation.calculate_energy_forces()
    np.testing.assert_allclose(system.positions, expected)


def test_fixed_slab_matches_manual_langevin(tmp_path):
    """Stationary plus disabled transforms preserves the manual MD trajectory."""
    manual = make_system()
    wrapped = make_system()
    manual_integrator = LangevinIntegrator(1., 300., 0.005)
    np.random.seed(10101)
    for _ in range(4):
        positions, velocities = manual_integrator.integrate(manual)
        positions[0] = manual.positions[0]
        velocities[0] = 0.
        manual.positions, manual.velocities = positions, velocities

    simulation = Simulation(
        wrapped, LangevinIntegrator(1., 300., 0.005),
        calculator=CompositeCalculator(wrapped, []),
        plugins=[Stationary('atom 0')], center=False, wrap=False,
        output_directory=str(tmp_path), dcd_write_interval=1,
    )
    try:
        np.random.seed(10101)
        simulation.run_dynamics(4)
        np.testing.assert_array_equal(wrapped.positions, manual.positions)
        np.testing.assert_array_equal(wrapped.velocities, manual.velocities)
        np.testing.assert_array_equal(wrapped.positions[0], [1., 2., 3.])
        for suffix in ('dcd', 'csv', 'log'):
            assert (tmp_path / f'pydft_qmmm.{suffix}').stat().st_size > 0
        assert not list(tmp_path.glob('*.npz'))
    finally:
        simulation.close()
