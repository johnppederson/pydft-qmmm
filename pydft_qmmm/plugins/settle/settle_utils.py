#! /usr/bin/env python3
"""A module containing the core functionality of the SETTLE algorithm.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def settle_positions(
        residues: list[list[int]],
        positions_i: NDArray[np.float64],
        positions_f: NDArray[np.float64],
        masses: NDArray[np.float64],
        oh_distance: int | float = 1.,
        hh_distance: int | float = 1.632981,
) -> NDArray[np.float64]:
    """A utility to perform the SETTLE algorithm on a set of positions.

    :param residues: The name of the water residues in the
        :class:`System`.
    :param positions_i: The initial positions of the :class:`System`, in
        Angstroms.
    :param positions_f: The final positions of the :class:`System`, in
        Angstroms.
    :param masses: |masses|
    :param oh_distance: The distance between the oxygen and hydrogen, in
        Angstroms.
    :param hh_distance: The distance between the hydrogens, in
        Angstroms.

    .. note:: Based on the SETTLE kernel in OpenMM.
    """
    pos_0 = positions_i[residues, :]
    oxy_0 = pos_0[:, 0, :]
    ha_0 = pos_0[:, 1, :]
    hb_0 = pos_0[:, 2, :]

    pos_1 = positions_f[residues, :]
    oxy_1 = pos_1[:, 0, :]
    ha_1 = pos_1[:, 1, :]
    hb_1 = pos_1[:, 2, :]

    mass = masses[[atom for res in residues for atom in res]].reshape((-1, 3))
    m_oxy = mass[0, 0]
    m_ha = mass[0, 1]
    m_hb = mass[0, 2]

    xp0 = oxy_1 - oxy_0
    xp1 = ha_1 - ha_0
    xp2 = hb_1 - hb_0

    xb0 = ha_0 - oxy_0
    xc0 = hb_0 - oxy_0

    xcom = ((xp0*m_oxy + (xb0+xp1)*m_ha + (xc0+xp2)*m_hb)
            / (m_oxy + m_ha + m_hb))

    xa1 = xp0 - xcom
    xb1 = xb0 + xp1 - xcom
    xc1 = xc0 + xp2 - xcom

    xaksZd = xb0[:, 1]*xc0[:, 2] - xb0[:, 2]*xc0[:, 1]
    yaksZd = xb0[:, 2]*xc0[:, 0] - xb0[:, 0]*xc0[:, 2]
    zaksZd = xb0[:, 0]*xc0[:, 1] - xb0[:, 1]*xc0[:, 0]

    xaksXd = xa1[:, 1]*zaksZd - xa1[:, 2]*yaksZd
    yaksXd = xa1[:, 2]*xaksZd - xa1[:, 0]*zaksZd
    zaksXd = xa1[:, 0]*yaksZd - xa1[:, 1]*xaksZd

    xaksYd = yaksZd*zaksXd - zaksZd*yaksXd
    yaksYd = zaksZd*xaksXd - xaksZd*zaksXd
    zaksYd = xaksZd*yaksXd - yaksZd*xaksXd

    axlng = (xaksXd**2 + yaksXd**2 + zaksXd**2)**0.5
    aylng = (xaksYd**2 + yaksYd**2 + zaksYd**2)**0.5
    azlng = (xaksZd**2 + yaksZd**2 + zaksZd**2)**0.5

    trns11 = xaksXd / axlng
    trns21 = yaksXd / axlng
    trns31 = zaksXd / axlng
    trns12 = xaksYd / aylng
    trns22 = yaksYd / aylng
    trns32 = zaksYd / aylng
    trns13 = xaksZd / azlng
    trns23 = yaksZd / azlng
    trns33 = zaksZd / azlng

    xb0d = trns11*xb0[:, 0] + trns21*xb0[:, 1] + trns31*xb0[:, 2]
    yb0d = trns12*xb0[:, 0] + trns22*xb0[:, 1] + trns32*xb0[:, 2]
    xc0d = trns11*xc0[:, 0] + trns21*xc0[:, 1] + trns31*xc0[:, 2]
    yc0d = trns12*xc0[:, 0] + trns22*xc0[:, 1] + trns32*xc0[:, 2]

    za1d = trns13*xa1[:, 0] + trns23*xa1[:, 1] + trns33*xa1[:, 2]

    xb1d = trns11*xb1[:, 0] + trns21*xb1[:, 1] + trns31*xb1[:, 2]
    yb1d = trns12*xb1[:, 0] + trns22*xb1[:, 1] + trns32*xb1[:, 2]
    zb1d = trns13*xb1[:, 0] + trns23*xb1[:, 1] + trns33*xb1[:, 2]
    xc1d = trns11*xc1[:, 0] + trns21*xc1[:, 1] + trns31*xc1[:, 2]
    yc1d = trns12*xc1[:, 0] + trns22*xc1[:, 1] + trns32*xc1[:, 2]
    zc1d = trns13*xc1[:, 0] + trns23*xc1[:, 1] + trns33*xc1[:, 2]

    rc = 0.5 * hh_distance
    rb = (oh_distance**2 - rc**2)**0.5
    ra = rb * (m_ha + m_hb) / (m_oxy + m_ha + m_hb)
    rb = rb - ra

    sinphi = za1d / ra
    cosphi = (1 - sinphi**2)**0.5
    sinpsi = (zb1d - zc1d) / (2*rc*cosphi)
    cospsi = (1 - sinpsi**2)**0.5

    ya2d = ra*cosphi
    xb2d = -rc*cospsi
    yb2d = -rb*cosphi - rc*sinpsi*sinphi
    yc2d = -rb*cosphi + rc*sinpsi*sinphi
    xb2d2 = xb2d**2
    hh2 = 4.0*xb2d2 + (yb2d-yc2d)**2 + (zb1d-zc1d)**2
    delx = 2.0*xb2d + (4.0*xb2d2 - hh2 + hh_distance**2)**0.5
    xb2d = xb2d - 0.5*delx

    alpha = xb2d*(xb0d-xc0d) + yb0d*yb2d + yc0d*yc2d
    beta = xb2d*(yc0d-yb0d) + xb0d*yb2d + xc0d*yc2d
    gamma = xb0d*yb1d - xb1d*yb0d + xc0d*yc1d - xc1d*yc0d

    al2be2 = alpha**2 + beta**2
    sintheta = (alpha*gamma - beta*(al2be2 - gamma**2)**0.5) / al2be2
    costheta = (1 - sintheta**2)**0.5
    xa3d = -ya2d*sintheta
    ya3d = ya2d*costheta
    za3d = za1d
    xb3d = xb2d*costheta - yb2d*sintheta
    yb3d = xb2d*sintheta + yb2d*costheta
    zb3d = zb1d
    xc3d = -xb2d*costheta - yc2d*sintheta
    yc3d = -xb2d*sintheta + yc2d*costheta
    zc3d = zc1d

    xa3 = trns11*xa3d + trns12*ya3d + trns13*za3d
    ya3 = trns21*xa3d + trns22*ya3d + trns23*za3d
    za3 = trns31*xa3d + trns32*ya3d + trns33*za3d
    xb3 = trns11*xb3d + trns12*yb3d + trns13*zb3d
    yb3 = trns21*xb3d + trns22*yb3d + trns23*zb3d
    zb3 = trns31*xb3d + trns32*yb3d + trns33*zb3d
    xc3 = trns11*xc3d + trns12*yc3d + trns13*zc3d
    yc3 = trns21*xc3d + trns22*yc3d + trns23*zc3d
    zc3 = trns31*xc3d + trns32*yc3d + trns33*zc3d

    xa3 = np.concatenate(
        (xa3.reshape(-1, 1), ya3.reshape(-1, 1), za3.reshape(-1, 1)),
        axis=1,
    )
    xb3 = np.concatenate(
        (xb3.reshape(-1, 1), yb3.reshape(-1, 1), zb3.reshape(-1, 1)),
        axis=1,
    )
    xc3 = np.concatenate(
        (xc3.reshape(-1, 1), yc3.reshape(-1, 1), zc3.reshape(-1, 1)),
        axis=1,
    )

    xp0 = xcom + xa3
    xp1 = xcom + xb3 - xb0
    xp2 = xcom + xc3 - xc0

    oxy_1 = oxy_0 + xp0
    ha_1 = ha_0 + xp1
    hb_1 = hb_0 + xp2

    pos_1 = np.concatenate(
        (
            oxy_1[:, np.newaxis, :],
            ha_1[:, np.newaxis, :],
            hb_1[:, np.newaxis, :],
        ),
        axis=1,
    )
    positions_f[residues, :] = pos_1.reshape(pos_0.shape)
    return positions_f


def settle_velocities(
        residues: list[list[int]],
        positions_i: NDArray[np.float64],
        velocities_i: NDArray[np.float64],
        masses: NDArray[np.float64],
) -> NDArray[np.float64]:
    """A utility to perform the SETTLE algorithm on a set of velocities.

    :param residues: The name of the water residues in the
        :class:`System`.
    :param positions_i: The initial positions of the :class:`System`, in
        Angstroms.
    :param velocities_f: The initial velocities of the :class:`System`,
        in Angstroms per femtosecond.
    :param masses: |masses|

    .. note:: Based on the SETTLE kernel in OpenMM.
    """
    pos_0 = positions_i[residues, :]
    oxy_p = pos_0[:, 0, :]
    ha_p = pos_0[:, 1, :]
    hb_p = pos_0[:, 2, :]

    vel_0 = velocities_i[residues, :]
    oxy_v = vel_0[:, 0, :]
    ha_v = vel_0[:, 1, :]
    hb_v = vel_0[:, 2, :]

    mass = masses[[atom for res in residues for atom in res]].reshape((-1, 3))
    m_oxy = mass[0, 0]
    m_ha = mass[0, 1]
    m_hb = mass[0, 2]

    eAB = ha_p - oxy_p
    eBC = hb_p - ha_p
    eCA = oxy_p - hb_p

    eAB = eAB * np.sum(eAB**2, axis=1, keepdims=True)**-0.5
    eBC = eBC * np.sum(eBC**2, axis=1, keepdims=True)**-0.5
    eCA = eCA * np.sum(eCA**2, axis=1, keepdims=True)**-0.5

    vAB = np.sum((ha_v - oxy_v)*eAB, axis=1, keepdims=True)
    vBC = np.sum((hb_v - ha_v)*eBC, axis=1, keepdims=True)
    vCA = np.sum((oxy_v - hb_v)*eCA, axis=1, keepdims=True)

    cA = -np.sum(eAB*eCA, axis=1, keepdims=True)
    cB = -np.sum(eAB*eBC, axis=1, keepdims=True)
    cC = -np.sum(eBC*eCA, axis=1, keepdims=True)

    s2A = 1 - cA**2
    s2B = 1 - cB**2
    s2C = 1 - cC**2

    mABCinv = 1 / (m_oxy*m_ha*m_hb)
    denom = ((
        (s2A*m_ha + s2B*m_oxy)*m_hb
        + (s2A*m_ha**2 + 2*(cA*cB*cC+1)*m_oxy*m_ha + s2B*m_oxy**2)
    )*m_hb + s2C*m_oxy*m_ha*(m_oxy+m_ha))*mABCinv
    tab = (
        (cB*cC*m_oxy - cA*(m_ha+m_hb))*vCA
        + (cA*cC*m_ha - cB*(m_hb+m_oxy))*vBC
        + (s2C*(m_oxy**2)*(m_ha**2)*mABCinv + (m_oxy+m_ha+m_hb))*vAB
    )/denom
    tbc = (
        (cA*cB*m_hb - cC*(m_ha+m_oxy))*vCA
        + (cA*cC*m_ha - cB*(m_hb+m_oxy))*vAB
        + (s2A*(m_ha**2)*(m_hb**2)*mABCinv + (m_oxy+m_ha+m_hb))*vBC
    )/denom
    tca = (
        (cA*cB*m_hb - cC*(m_ha+m_oxy))*vBC
        + (cB*cC*m_oxy - cA*(m_ha+m_hb))*vAB
        + (s2B*(m_oxy**2)*(m_hb**2)*mABCinv + (m_oxy+m_ha+m_hb))*vCA
    )/denom

    oxy_v = oxy_v + (eAB*tab - eCA*tca)/m_oxy
    ha_v = ha_v + (eBC*tbc - eAB*tab)/m_ha
    hb_v = hb_v + (eCA*tca - eBC*tbc)/m_hb

    vel_1 = np.concatenate(
        (
            oxy_v[:, np.newaxis, :],
            ha_v[:, np.newaxis, :],
            hb_v[:, np.newaxis, :],
        ),
        axis=1,
    )
    velocities_f = velocities_i
    velocities_f[residues, :] = vel_1.reshape(vel_0.shape)
    return velocities_f
