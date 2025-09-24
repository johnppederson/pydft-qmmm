"""Helper functions for operations involving periodic lattices.
"""
from __future__ import annotations

__all__ = [
    "compute_least_mirror",
    "compute_lattice_constants",
    "compute_lattice_vectors",
]

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def compute_least_mirror(
        i_vector: NDArray[np.float64],
        j_vector: NDArray[np.float64],
        box: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Calculate the least mirror vector.

    Args:
        i_vector: The position vector (:math:`\mathrm{\mathring{A}}`).
        j_vector: The reference vector (:math:`\mathrm{\mathring{A}}`).
        box: The lattice vectors (:math:`\mathrm{\mathring{A}}`) of an
            arbitrary triclinic box.

    Returns:
        Returns the least mirror coordinates of i_vector with respect to
        j_vector given a set of lattice vectors from a periodic
        triclinic system.
    """
    r_vector = i_vector - j_vector
    r_vector -= box[2] * np.floor(r_vector[2]/box[2][2] + 0.5)
    r_vector -= box[1] * np.floor(r_vector[1]/box[1][1] + 0.5)
    r_vector -= box[0] * np.floor(r_vector[0]/box[0][0] + 0.5)
    return r_vector


def compute_lattice_constants(
        box: NDArray[np.float64],
) -> tuple[float, ...]:
    r"""Calculate the length and angle constants from lattice vectors.

    Returns the lattice constants a, b, c, |alpha|, |beta|, and |gamma|
    using a set of box vectors for a periodic triclinic system.

    Args:
        box: The lattice vectors (:math:`\mathrm{\mathring{A}}`) of an
            arbitrary triclinic box.

    Returns:
        The characteristic lengths (:math:`\mathrm{\mathring{A}}`) and angles
        (:math:`\mathrm{^{\circ}}`) of an arbitrary triclinic box.
    """
    vec_a = box[:, 0]
    vec_b = box[:, 1]
    vec_c = box[:, 2]
    len_a = np.linalg.norm(vec_a)
    len_b = np.linalg.norm(vec_b)
    len_c = np.linalg.norm(vec_c)
    alpha = 180*np.arccos(np.dot(vec_b, vec_c)/len_b/len_c)/np.pi
    beta = 180*np.arccos(np.dot(vec_a, vec_c)/len_a/len_c)/np.pi
    gamma = 180*np.arccos(np.dot(vec_a, vec_b)/len_a/len_b)/np.pi
    return tuple(float(x) for x in (len_a, len_b, len_c, alpha, beta, gamma))


def compute_lattice_vectors(
        a: float,
        b: float,
        c: float,
        alpha: float,
        beta: float,
        gamma: float,
) -> NDArray[np.float64]:
    r"""Calculate the lattice vectors from length and angle constants.

    Args:
        a: The first characteristic length
            (:math:`\mathrm{\mathring{A}}`) of an arbitrary triclinic
            box.
        b: The second characteristic length
            (:math:`\mathrm{\mathring{A}}`) of an arbitrary triclinic
            box.
        b: The third characteristic length
            (:math:`\mathrm{\mathring{A}}`) of an arbitrary triclinic
            box.
        alpha: The first characteristic angle (:math:`\mathrm{^{\circ}}`),
            |alpha|, of an arbitrary triclinic box.
        beta: The second characteristic angle (:math:`\mathrm{^{\circ}}`),
            |beta|, of an arbitrary triclinic box.
        gamma: The third characteristic angle (:math:`\mathrm{^{\circ}}`),
            |gamma|, of an arbitrary triclinic box.

    Returns:
        The lattice vectors (:math:`\mathrm{\mathring{A}}`) of an
        arbitrary triclinic box.
    """
    alpha *= np.pi/180
    beta *= np.pi/180
    gamma *= np.pi/180
    vec_a = np.array([[a], [0.], [0.]])
    vec_b = np.array(
        [[b*np.cos(gamma)],
         [b*np.sin(gamma)],
         [0.]],
    )
    c_y = (np.cos(alpha) - np.cos(beta)*np.cos(gamma))/np.sin(gamma)
    vec_c = np.array(
        [[c*np.cos(beta)],
         [c*c_y],
         [np.sqrt(c**2 - (c*np.cos(beta))**2 - (c*c_y)**2)]],
    )
    box = np.concatenate((vec_a, vec_b, vec_c), axis=1)
    box[box**2 < 1e-12] = 0.
    box[:, 2] -= box[:, 1]*np.round(box[1, 2]/box[1, 1])
    box[:, 2] -= box[:, 0]*np.round(box[0, 2]/box[0, 0])
    box[:, 1] -= box[:, 0]*np.round(box[0, 1]/box[0, 0])
    return box
