"""Arbitrary-point PME for explicit grids, exclusions, and unequal cells.

These operators remain necessary for partitioned and molecular comparisons.
The native grid dispatcher uses them only when direct grid evaluation is
not applicable.
"""
import numpy as np


def point_potential(positions, charges, excluded, alpha, gridnumber,
                    spline_order, shape, cell, chunk, source_cell, origin):
    """Evaluate a reciprocal field at arbitrary target grid coordinates."""
    import helpme_py
    from .pme_grid import (
        COULOMB_CONSTANT, KJMOL_PER_EV, _lattice_constants,
        _minimum_image, grid_coordinates,
    )
    pme = helpme_py.PMEInstanceD()
    pme.setup(1, alpha, spline_order, *gridnumber, COULOMB_CONSTANT, 0)
    pme.set_lattice_vectors(
        *_lattice_constants(np.asarray(
            cell if source_cell is None else source_cell, dtype=np.float64,
        )),
        helpme_py.LatticeType.XAligned,
    )
    all_charges = helpme_py.MatrixD(charges.reshape(-1, 1))
    all_positions = helpme_py.MatrixD(positions)
    if excluded.size:
        excluded_charges = helpme_py.MatrixD(charges[excluded].reshape(-1, 1))
        excluded_positions = helpme_py.MatrixD(positions[excluded, :])

    coordinates = grid_coordinates(shape, cell)
    if origin is not None:
        coordinates += np.asarray(origin)
    potential = np.zeros(len(coordinates))
    for start in range(0, len(coordinates), chunk):
        block = np.ascontiguousarray(coordinates[start:start + chunk])
        values = np.zeros((len(block), 1))
        matrix = helpme_py.MatrixD(values)
        pme.compute_P_rec(
            0, all_charges, all_positions,
            helpme_py.MatrixD(block), 0, matrix,
        )
        if excluded.size:
            pme.compute_P_adj(
                0, excluded_charges, excluded_positions,
                helpme_py.MatrixD(block), matrix,
                _minimum_image(),
            )
        potential[start:start + chunk] = values[:, 0]
    # helPME reports the electrostatic potential in kJ/mol/e; VASP wants
    # the electron potential ENERGY in eV, hence the negation.
    return (-potential / KJMOL_PER_EV).reshape(shape)


def point_source_forces(positions, charges, excluded, alpha, gridnumber,
                        spline_order, grid_charges, shape, cell, points,
                        point_charges, source_cell, origin):
    """Transpose the arbitrary-point field, including exclusion terms."""
    import helpme_py
    from .pme_grid import (
        COULOMB_CONSTANT, _lattice_constants,
        _minimum_image, grid_coordinates,
    )
    pme = helpme_py.PMEInstanceD()
    pme.setup(1, alpha, spline_order, *gridnumber, COULOMB_CONSTANT, 0)
    pme.set_lattice_vectors(
        *_lattice_constants(np.asarray(
            cell if source_cell is None else source_cell, dtype=np.float64,
        )),
        helpme_py.LatticeType.XAligned,
    )
    sources = np.asarray(grid_charges, dtype=np.float64).reshape(-1)
    coordinates = grid_coordinates(shape, cell)
    if origin is not None:
        coordinates += np.asarray(origin)
    if points is not None and len(points):
        coordinates = np.vstack(
            [coordinates, np.asarray(points).reshape(-1, 3)],
        )
        sources = np.concatenate(
            [sources, np.asarray(point_charges).reshape(-1)],
        )
    sources = helpme_py.MatrixD(np.ascontiguousarray(sources.reshape(-1, 1)))
    coordinates = helpme_py.MatrixD(np.ascontiguousarray(coordinates))
    forces = np.zeros((len(charges), 3))
    targets = np.flatnonzero(charges)
    if targets.size:
        field = np.zeros((targets.size, 4))
        pme.compute_P_rec(
            0, sources, coordinates,
            helpme_py.MatrixD(np.ascontiguousarray(positions[targets])),
            1, helpme_py.MatrixD(field),
        )
        forces[targets] = -field[:, 1:] * charges[targets, None]
    if excluded.size:
        # The transpose of the erf subtraction in build_pme_potential.
        adjustment = np.zeros((excluded.size, 4))
        pme.compute_PDP_adj(
            0, sources, coordinates,
            helpme_py.MatrixD(np.ascontiguousarray(positions[excluded])),
            helpme_py.MatrixD(adjustment), _minimum_image(),
        )
        forces[excluded] -= adjustment[:, 1:] * charges[excluded, None]
    return forces
