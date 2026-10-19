"""Reusable initialization and engine lifecycle checks without external engines."""
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pytest

from pydft_qmmm import Atom, Simulation, System, VerletIntegrator, generate_velocities
from pydft_qmmm.calculators import Calculator, CompositeCalculator, PotentialCalculator, Results
from pydft_qmmm.utils import KB


def test_frozen_velocities_match_previous_initialization():
    masses = np.array([1., 196.966569, 15.99943, 1.007947])
    old = generate_velocities(masses, 300, 10101)
    old[[0, 1]] = 0.
    kinetic = .5e4 * np.sum(masses[2:, None] * old[2:]**2)
    old *= np.sqrt((.5 * 3 * 2 * KB / 1000 * 300) / kinetic)
    expected_next_random = np.random.standard_normal(6)
    new = generate_velocities(masses, 300, 10101, frozen=[0, 1], rescale=True)
    np.testing.assert_allclose(new, old, rtol=1e-15, atol=1e-18)
    np.testing.assert_array_equal(np.random.standard_normal(6), expected_next_random)
    assert .5e4 * np.sum(masses[:, None] * new**2) == pytest.approx(3 * KB / 1000 * 300)


def test_zero_mass_and_zero_seed_velocities():
    with np.errstate(divide='raise', invalid='raise'):
        first = generate_velocities(np.array([0., 1., 2.]), 300, 0, rescale=True)
        second = generate_velocities(np.array([0., 1., 2.]), 300, 0, rescale=True)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[0], 0.)
    with pytest.raises(ValueError, match='mobile'):
        generate_velocities(np.array([1.]), 300, frozen=[0], rescale=True)
    with pytest.raises(ValueError, match='indices'):
        generate_velocities(np.array([1.]), 300, frozen=[1])


@dataclass(frozen=True)
class FailingCalculator(Calculator):
    closed: list = field(default_factory=list)

    @property
    def name(self):
        return "Failing"

    def calculate(self, return_forces=True, return_components=True):
        raise RuntimeError('SCF failed')

    def close(self):
        self.closed.append(True)


def test_simulation_closes_engines_on_error(tmp_path):
    system = System([Atom(element='H', mass=1.)])
    failed = FailingCalculator(system)
    with pytest.raises(RuntimeError, match='SCF failed'):
        with Simulation(system, VerletIntegrator(1), calculator=failed,
                        output_directory=str(tmp_path)) as simulation:
            simulation.run_dynamics(1)
    assert failed.closed == [True]
    assert all(handler.stream is None or handler.stream.closed
               for handler in simulation.handlers)
    simulation.close()
    assert failed.closed == [True]


def test_composite_closes_all_potentials_even_on_failure():
    system = System([Atom(element='H', mass=1.)])
    closed = []

    def fail():
        closed.append('failed')
        raise RuntimeError('cleanup failed')

    calculator = CompositeCalculator(system, [
        PotentialCalculator(system, SimpleNamespace(close=lambda: closed.append('ok'))),
        PotentialCalculator(system, SimpleNamespace(close=fail)),
    ])
    with pytest.raises(RuntimeError, match='cleanup failed'):
        calculator.close()
    assert sorted(closed) == ['failed', 'ok']


def test_simulations_do_not_share_log_handlers(tmp_path):
    system = System([Atom(element='H', mass=1.)])
    with Simulation(system, VerletIntegrator(1), calculator=CompositeCalculator(system, []),
                    output_directory=str(tmp_path / 'a')) as a:
        with Simulation(system, VerletIntegrator(1), calculator=CompositeCalculator(system, []),
                        output_directory=str(tmp_path / 'b')) as b:
            a.run_dynamics(0)
            assert (tmp_path / 'a/pydft_qmmm.csv').stat().st_size > 0
            assert (tmp_path / 'b/pydft_qmmm.csv').stat().st_size == 0
            b.run_dynamics(0)
