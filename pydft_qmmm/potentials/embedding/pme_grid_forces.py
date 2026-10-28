"""Transpose helPME's native grid potential without spreading grid charges.

The order, mesh offset, spline moduli and grid-node probing factor match
helPME 0.3.1's conventional Coulomb PME operator. Only positive diagonal
cells are handled here; the general point evaluator remains the fallback.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

# helPME's makeBSplines and makeGridProbeFactors use a float constant.
_OFFSET = float(np.float32(1e-6))


def _weights(fraction, order):
    """Cardinal B-spline weights on helPME's forward stencil."""
    fraction = np.asarray(fraction, dtype=float)
    weights = np.zeros((*fraction.shape, order))
    weights[..., 0] = 1 - fraction
    weights[..., 1] = fraction
    for size in range(3, order + 1):
        weights[..., size - 1] = fraction * weights[..., size - 2] / (size - 1)
        for index in range(size - 2, 0, -1):
            weights[..., index] = (
                (fraction + size - index - 1) * weights[..., index - 1]
                + (index + 1 - fraction) * weights[..., index]
            ) / (size - 1)
        weights[..., 0] *= (1 - fraction) / (size - 1)
    return weights


def _axis_factor(size, order):
    """Inverse spline modulus times the transposed grid-node probe."""
    phase = 2 * np.pi * np.arange(size)[:, None] * np.arange(order) / size
    transform = np.exp(1j * phase) @ _weights(0., order)
    moduli = np.abs(transform)**2
    # Match helPME's small-modulus repair, including odd-order Nyquist modes.
    if moduli[0] < 1e-7:
        moduli[0] = moduli[1] / 2
    for index in range(size - 1):
        if moduli[index] < 1e-7:
            moduli[index] = (moduli[index - 1] + moduli[index + 1]) / 2
    if moduli[-1] < 1e-7:
        moduli[-1] = moduli[-2] / 2
    phase -= 2 * np.pi * np.arange(size)[:, None] / size
    probe = np.exp(-1j * phase) @ _weights(1 - _OFFSET * size, order)
    return probe / moduli


@lru_cache(maxsize=1)
def _transpose_kernel(shape, lengths, alpha, order, coulomb):
    """Cached real-FFT kernel in kJ/mol/e per mesh charge (e)."""
    frequencies = [
        np.fft.fftfreq(n) * n / length for n, length in zip(shape, lengths)
    ]
    frequencies[2] = np.fft.rfftfreq(shape[2]) * shape[2] / lengths[2]
    norm = (frequencies[0][:, None, None]**2
            + frequencies[1][None, :, None]**2
            + frequencies[2][None, None, :]**2)
    norm[0, 0, 0] = 1.
    kernel = np.exp(-np.pi**2 * norm / alpha**2) / norm
    # NumPy normalizes its inverse FFT; helPME's FFTW inverse is unnormalized.
    kernel *= np.prod(shape) * coulomb / (np.pi * np.prod(lengths))
    kernel[0, 0, 0] = 0.
    kernel = kernel.astype(np.complex128)
    for axis, size in enumerate(shape):
        factor = _axis_factor(size, order)
        if axis == 2:
            factor = factor[:size // 2 + 1]
        dimensions = [1, 1, 1]
        dimensions[axis] = len(factor)
        kernel *= factor.reshape(dimensions)
    kernel.setflags(write=False)
    return kernel


def native_grid_source_forces(
        positions, charges, grid_charges, shape, lengths, alpha, order, coulomb,
):
    """Differentiate the native-grid energy with respect to MM positions.

    This is the transpose of grid probing and reciprocal convolution,
    followed by derivatives of the original charge-spreading weights.
    It deliberately does not differentiate a continuum approximation to PME.
    """
    grid_charges = np.asarray(grid_charges, dtype=float)
    if grid_charges.shape != tuple(shape) or not np.isfinite(grid_charges).all():
        raise ValueError("Invalid QM grid charges for native PME forces")
    spectrum = np.fft.rfftn(grid_charges)
    spectrum *= _transpose_kernel(
        tuple(shape), tuple(lengths), alpha, order, coulomb,
    )
    potential = np.fft.irfftn(spectrum, s=shape, axes=(0, 1, 2))
    forces = np.zeros((len(charges), 3))
    targets = np.flatnonzero(charges)
    scale = np.asarray(shape) / lengths
    for start in range(0, len(targets), 512):
        atoms = targets[start:start + 512]
        scaled = ((positions[atoms] / lengths - _OFFSET) % 1.) * shape
        base = np.floor(scaled).astype(np.int64)
        fraction = scaled - base
        indices = (
            (base[..., None] + np.arange(order))
            % np.asarray(shape)[None, :, None]
        )
        weights = _weights(fraction, order)
        lower = _weights(fraction, order - 1)
        derivatives = np.zeros_like(weights)
        derivatives[..., :-1] -= lower
        derivatives[..., 1:] += lower
        values = potential[
            indices[:, 0, :, None, None],
            indices[:, 1, None, :, None],
            indices[:, 2, None, None, :],
        ]
        for axis in range(3):
            factors = [
                derivatives[:, a] if a == axis else weights[:, a]
                for a in range(3)
            ]
            gradient = np.einsum(
                'nijk,ni,nj,nk->n', values, *factors, optimize=True,
            )
            forces[atoms, axis] = -charges[atoms] * scale[axis] * gradient
    return forces
