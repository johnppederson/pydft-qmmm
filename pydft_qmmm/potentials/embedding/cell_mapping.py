"""Aligned restriction of a periodic MM grid to a smaller QM cell."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class CellMapping:
    """Cells use row vectors; origin and positions are Cartesian Angstrom."""

    def __init__(self, mm_cell, qm_cell, origin, mm_shape, qm_shape):
        self.mm_cell = np.asarray(mm_cell, dtype=float)
        self.qm_cell = np.asarray(qm_cell, dtype=float)
        self.origin = np.asarray(origin, dtype=float)
        for cell in (self.mm_cell, self.qm_cell):
            if (
                cell.shape != (3, 3) or not np.isfinite(cell).all()
                or not np.allclose(cell, np.diag(np.diag(cell)))
                or np.any(np.diag(cell) <= 0)
            ):
                raise ValueError(
                    "Separate cells must be positive orthorhombic lattices",
                )
        if (
            self.origin.shape != (3,) or not np.isfinite(self.origin).all()
            or not np.allclose(self.origin[:2], 0)
        ):
            raise ValueError("Only a finite z origin is supported")
        if (
            not np.allclose(self.mm_cell[:2], self.qm_cell[:2])
            or self.qm_cell[2, 2] > self.mm_cell[2, 2] + 1e-8
        ):
            raise ValueError("Only shortening the z lattice is supported")
        shapes = []
        for shape in (mm_shape, qm_shape):
            a = np.asarray(shape)
            if (
                    a.shape != (3,) or not np.isfinite(a).all()
                    or np.any(a < 1) or np.any(a != np.rint(a))
            ):
                raise ValueError("Grid dimensions must be positive integers")
            shapes.append(tuple(int(n) for n in a))
        self.mm_shape, self.qm_shape = shapes
        spacing = np.diag(self.mm_cell) / self.mm_shape
        qm_spacing = np.diag(self.qm_cell) / self.qm_shape
        if not np.allclose(spacing, qm_spacing, rtol=0, atol=1e-8):
            raise ValueError("QM and MM grid spacings must match")
        offset = self.origin / spacing
        if not np.allclose(offset, np.rint(offset), rtol=0, atol=1e-7):
            raise ValueError("QM origin must align with the MM grid")
        self.indices = np.ix_(
            *[
                (np.arange(n) + int(o)) % m
                for n, o, m in zip(
                    self.qm_shape, np.rint(offset), self.mm_shape,
                )
            ],
        )

    def validate_positions(self, positions):
        local = np.asarray(positions) - self.origin
        if (
                not np.isfinite(local).all()
                or np.any(local[:, 2] <= 0)
                or np.any(local[:, 2] >= self.qm_cell[2, 2])
        ):
            raise ValueError(
                "QM atoms must lie strictly inside the QM z faces",
            )

    def restrict(self, values):
        if np.shape(values) != self.mm_shape:
            raise ValueError("Incorrect MM grid shape")
        return np.asarray(values)[self.indices].copy()

    def transpose(self, charges):
        if np.shape(charges) != self.qm_shape:
            raise ValueError("Incorrect QM grid shape")
        result = np.zeros(self.mm_shape)
        result[self.indices] = charges
        return result

    def metadata(self):
        return dict(
            version=1, mm_cell=self.mm_cell.tolist(),
            qm_cell=self.qm_cell.tolist(), origin=self.origin.tolist(),
            mm_shape=self.mm_shape, qm_shape=self.qm_shape,
        )


def read_mapping(shape, cell, path="QM_CELL.json"):
    """Load explicit cell metadata and verify the actual VASP callback grid."""
    if not Path(path).exists():
        return None
    data = json.loads(Path(path).read_text())
    if data.pop("version") != 1:
        raise ValueError("Unsupported QM cell metadata version")
    mapping = CellMapping(**data)
    if (
            tuple(shape) != mapping.qm_shape
            or not np.allclose(cell, mapping.qm_cell, rtol=0, atol=1e-7)
    ):
        raise ValueError(
            "VASP callback cell/grid differs from the requested QM grid",
        )
    return mapping
