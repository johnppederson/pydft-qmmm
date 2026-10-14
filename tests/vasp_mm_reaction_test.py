"""MM reactions must differentiate the discrete VASP embedding energy."""
from types import SimpleNamespace

import numpy as np
import pytest

from pydft_qmmm.embedding.grid_potential import (
    build_external_potential, nuclear_charge_on_grid,
    spectral_value_and_gradient, spline_value_and_gradient,
)
from pydft_qmmm.interfaces.vasp import vasp_plugin


@pytest.mark.parametrize('method', ['spline', 'spectral'])
@pytest.mark.parametrize('shape', [(15, 17, 19), (16, 18, 20)])
def test_nuclear_reaction_transposes_energy_interpolation(method, shape):
    cell = np.array([[8., 0., 0.], [1., 9., 0.], [.5, -.3, 10.]])
    points = np.array([[.12, .29, .31], [.87, .61, .93]]) @ cell
    charges = np.array([6., 1.])
    field = np.random.default_rng(19).normal(size=shape)
    interpolate = spline_value_and_gradient if method == 'spline' else spectral_value_and_gradient
    values, _ = interpolate(field, cell, points)
    grid_charge = nuclear_charge_on_grid(shape, cell, points, charges, method)
    np.testing.assert_allclose(np.sum(grid_charge * field), charges @ values, atol=1e-12)
    np.testing.assert_allclose(grid_charge.sum(), charges.sum(), atol=1e-12)


@pytest.mark.parametrize('method', ['spline', 'spectral'])
@pytest.mark.parametrize('shape', [(15, 17, 19), (16, 18, 20)])
def test_callback_mm_force_is_full_embedding_energy_derivative(method, shape, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(vasp_plugin.NUCLEAR_FIELD_ENV, method)
    monkeypatch.delenv(vasp_plugin.ERFC_NEAR_ENV, raising=False)
    monkeypatch.delenv(vasp_plugin.COARSE_NEAR_ENV, raising=False)
    cell = np.array([[8., 0., 0.], [1., 9., 0.], [.5, -.3, 10.]])
    qm_fractional = np.array([[.25, .3, .4], [.4, .31, .47]])
    qm_positions = qm_fractional @ cell
    valence = np.array([6., 1.])
    mm_positions = np.array([[.54, .51, .55], [.67, .52, .57]]) @ cell
    mm_charges = np.array([-.8, .4])
    sigma = .4
    density = np.random.default_rng(11).uniform(.1, 1., size=shape)
    density *= valence.sum() / density.mean()
    constants = SimpleNamespace(lattice_vectors=cell, shape_grid=shape,
                               positions=qm_fractional, ZVAL=valence, ion_types=[0, 1],
                               forces=np.zeros((2, 3)), charge_density=density)
    # Deliberately unrelated cached fields: forces must use the final density
    # and the nuclear energy interpolation, not these SCF/pseudopotential fields.
    monkeypatch.setattr(vasp_plugin, '_CACHE', {'hartree': np.ones(shape)*123.,
                                              'ion': np.random.default_rng(8).normal(size=shape)})
    monkeypatch.setattr(vasp_plugin, 'read_mm_charges',
                        lambda _: (mm_positions, mm_charges, 7, sigma))
    monkeypatch.setattr(vasp_plugin, '_external_potential',
                        lambda _: build_external_potential(mm_positions, mm_charges, shape, cell, sigma))
    additions = SimpleNamespace(total_energy=0., forces=np.zeros((2, 3)))
    vasp_plugin.force_and_stress(constants, additions)
    actual = np.loadtxt(vasp_plugin.FORCE_FILE, skiprows=1)
    interpolate = spline_value_and_gradient if method == 'spline' else spectral_value_and_gradient

    def energy():
        field = build_external_potential(mm_positions, mm_charges, shape, cell, sigma)
        values, _ = interpolate(field, cell, qm_positions)
        return (np.mean(density * field) - valence @ values) * vasp_plugin.KJMOL_PER_EV

    step = 1e-5
    for atom in range(2):
        for axis in range(3):
            initial = mm_positions[atom, axis]
            mm_positions[atom, axis] = initial + step
            plus = energy()
            mm_positions[atom, axis] = initial - step
            minus = energy()
            mm_positions[atom, axis] = initial
            np.testing.assert_allclose(actual[atom, axis], -(plus-minus)/(2*step),
                                       atol=2e-5, rtol=2e-6)
