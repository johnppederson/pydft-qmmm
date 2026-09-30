"""Accuracy and energy-gradient checks for periodic nuclear embedding."""
import numpy as np
import pytest
from types import SimpleNamespace

from pydft_qmmm.embedding.grid_potential import spline_value_and_gradient
from pydft_qmmm.embedding.grid_potential import spectral_value_and_gradient
from pydft_qmmm.interfaces.vasp import vasp_plugin


@pytest.mark.parametrize("shape", [(16, 18, 20), (17, 19, 21)])
def test_spline_reproduces_samples_and_periodic_images(shape):
    rng = np.random.default_rng(17)
    field = rng.normal(size=shape)
    cell = np.array([[8., 0., 0.], [1., 9., 0.], [.5, -.3, 10.]])
    indices = np.array([[0, 0, 0], [2, 7, 11], np.array(shape) - 1])
    points = indices / np.array(shape) @ cell
    values, gradients = spline_value_and_gradient(field, cell, points)
    np.testing.assert_allclose(values, field[tuple(indices.T)], atol=1e-12)
    shifted = points + np.array([2, -1, 3]) @ cell
    other_values, other_gradients = spline_value_and_gradient(field, cell, shifted)
    np.testing.assert_allclose(other_values, values, atol=1e-12)
    np.testing.assert_allclose(other_gradients, gradients, atol=1e-11)


def test_spline_gradient_is_energy_derivative_in_skew_cell():
    rng = np.random.default_rng(19)
    shape = (15, 18, 21)
    field = rng.normal(size=shape)
    cell = np.array([[8., 0., 0.], [1., 9., 0.], [.5, -.3, 10.]])
    points = np.array([[0., 0., 0.], [7.99999, -.01, 10.1], [2.1, 3.4, 5.6]])
    _, gradients = spline_value_and_gradient(field, cell, points)
    for axis in range(3):
        shift = np.eye(3)[axis] * 1e-5
        plus, _ = spline_value_and_gradient(field, cell, points + shift)
        minus, _ = spline_value_and_gradient(field, cell, points - shift)
        np.testing.assert_allclose((plus - minus) / 2e-5, gradients[:, axis],
                                   rtol=1e-7, atol=1e-8)


def test_spline_converges_to_known_fourier_field():
    cell = np.array([[8., 0., 0.], [1., 9., 0.], [.5, -.3, 10.]])
    fractional = np.array([[.123, .456, .789], [.9999, .0001, .51]])
    mode = np.array([1, -2, 1])
    phase = 2 * np.pi * fractional @ mode
    exact_value = np.cos(phase)
    exact_gradient = -np.sin(phase)[:, None] * (2 * np.pi * np.linalg.inv(cell) @ mode)
    errors = []
    for n in (16, 32):
        mesh = np.stack(np.meshgrid(*([np.arange(n) / n] * 3), indexing="ij"), -1)
        field = np.cos(2 * np.pi * mesh @ mode)
        values, gradients = spline_value_and_gradient(field, cell, fractional @ cell)
        errors.append(np.max(np.abs(gradients - exact_gradient)))
        if n == 32:
            np.testing.assert_allclose(values, exact_value, atol=1e-6)
            np.testing.assert_allclose(gradients, exact_gradient, atol=1e-5)
    assert errors[1] < errors[0] / 10


def test_empty_targets():
    values, gradients = spline_value_and_gradient(np.zeros((8, 8, 8)), np.eye(3), [])
    assert values.shape == (0,)
    assert gradients.shape == (0, 3)


@pytest.mark.parametrize("route", [None, "spectral"])
def test_force_callback_uses_selected_field_and_valence(route, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    if route is None:
        monkeypatch.delenv(vasp_plugin.NUCLEAR_FIELD_ENV, raising=False)
    else:
        monkeypatch.setenv(vasp_plugin.NUCLEAR_FIELD_ENV, route)
    shape = (8, 8, 8)
    cell = np.diag([8., 9., 10.])
    field = np.random.default_rng(21).normal(size=shape)
    fractional = np.array([[.17, .31, .65], [.93, .51, .07]])
    constants = SimpleNamespace(lattice_vectors=cell, shape_grid=shape,
                               positions=fractional, ZVAL=[11., 6.],
                               ion_types=[0, 1], forces=np.zeros((2, 3)))
    additions = SimpleNamespace(total_energy=0., forces=np.zeros((2, 3)))
    monkeypatch.setattr(vasp_plugin, "_external_potential", lambda _: field)
    monkeypatch.setattr(vasp_plugin, "_CACHE", {
        "hartree": np.zeros(shape), "ion": np.zeros(shape),
    })
    (tmp_path / vasp_plugin.CHARGE_FILE).write_text("0 0 0.3\n")
    evaluate = spectral_value_and_gradient if route else spline_value_and_gradient
    values, gradients = evaluate(field, cell, fractional @ cell)
    vasp_plugin.force_and_stress(constants, additions)
    np.testing.assert_allclose(additions.total_energy, -np.dot([11., 6.], values))
    np.testing.assert_allclose(additions.forces, gradients * np.array([[11.], [6.]]))
