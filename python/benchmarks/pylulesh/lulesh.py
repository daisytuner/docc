# See LICENSE.md for full license.
"""
This file contains the computational core of PyLULESH.
"""

import numpy as np
from constants import *
from domain import Domain, intarr, realarr, RealT
from typing import Dict, Tuple
import util

ptiny = 1e-36
gamma = np.array(
    [
        [1, 1, -1, -1, -1, -1, 1, 1],
        [1, -1, -1, 1, -1, 1, 1, -1],
        [1, -1, 1, -1, 1, -1, 1, -1],
        [-1, 1, -1, 1, 1, -1, 1, -1],
    ],
    dtype=RealT,
)


def time_increment(domain: Domain):
    """
    Advance time and set time increment.
    """
    # Compute new delta-time as necessary
    targetdt: float = domain.stoptime - domain.time
    if domain.dtfixed <= 0.0 and domain.cycle != 0:
        olddt = domain.deltatime

        newdt = 1e20
        if domain.dtcourant < newdt:
            newdt = domain.dtcourant / 2
        if domain.dthydro < newdt:
            newdt = domain.dthydro * 2 / 3

        ratio = newdt / olddt
        if ratio >= 1:
            if ratio < domain.deltatimemultlb:
                newdt = olddt
            elif ratio > domain.deltatimemultub:
                newdt = olddt * domain.deltatimemultub

        if newdt > domain.dtmax:
            newdt = domain.dtmax

        domain.deltatime = newdt

    # Try to prevent very small scaling on next iteration
    if targetdt > domain.deltatime and (targetdt < (4 * domain.deltatime / 3)):
        targetdt = 2 * domain.deltatime / 3

    if targetdt < domain.deltatime:
        domain.deltatime = targetdt

    # Increment
    domain.time += domain.deltatime
    domain.cycle += 1


def collect_domain_nodes_to_elem_nodes(domain: Domain, elem_to_node: intarr):
    return (domain.x[elem_to_node], domain.y[elem_to_node], domain.z[elem_to_node])


def init_stress_terms_for_elems(domain: Domain):
    sigxx = -domain.p - domain.q
    return sigxx, np.copy(sigxx), np.copy(sigxx)


def calc_elem_shape_function_derivatives(
    x: realarr, y: realarr, z: realarr
) -> Tuple[realarr, realarr]:
    x0 = x[:, 0]
    x1 = x[:, 1]
    x2 = x[:, 2]
    x3 = x[:, 3]
    x4 = x[:, 4]
    x5 = x[:, 5]
    x6 = x[:, 6]
    x7 = x[:, 7]
    y0 = y[:, 0]
    y1 = y[:, 1]
    y2 = y[:, 2]
    y3 = y[:, 3]
    y4 = y[:, 4]
    y5 = y[:, 5]
    y6 = y[:, 6]
    y7 = y[:, 7]
    z0 = z[:, 0]
    z1 = z[:, 1]
    z2 = z[:, 2]
    z3 = z[:, 3]
    z4 = z[:, 4]
    z5 = z[:, 5]
    z6 = z[:, 6]
    z7 = z[:, 7]

    fjxxi = 0.125 * ((x6 - x0) + (x5 - x3) - (x7 - x1) - (x4 - x2))
    fjxet = 0.125 * ((x6 - x0) - (x5 - x3) + (x7 - x1) - (x4 - x2))
    fjxze = 0.125 * ((x6 - x0) + (x5 - x3) + (x7 - x1) + (x4 - x2))

    fjyxi = 0.125 * ((y6 - y0) + (y5 - y3) - (y7 - y1) - (y4 - y2))
    fjyet = 0.125 * ((y6 - y0) - (y5 - y3) + (y7 - y1) - (y4 - y2))
    fjyze = 0.125 * ((y6 - y0) + (y5 - y3) + (y7 - y1) + (y4 - y2))

    fjzxi = 0.125 * ((z6 - z0) + (z5 - z3) - (z7 - z1) - (z4 - z2))
    fjzet = 0.125 * ((z6 - z0) - (z5 - z3) + (z7 - z1) - (z4 - z2))
    fjzze = 0.125 * ((z6 - z0) + (z5 - z3) + (z7 - z1) + (z4 - z2))

    # Compute cofactors
    cjxxi = (fjyet * fjzze) - (fjzet * fjyze)
    cjxet = -(fjyxi * fjzze) + (fjzxi * fjyze)
    cjxze = (fjyxi * fjzet) - (fjzxi * fjyet)

    cjyxi = -(fjxet * fjzze) + (fjzet * fjxze)
    cjyet = (fjxxi * fjzze) - (fjzxi * fjxze)
    cjyze = -(fjxxi * fjzet) + (fjzxi * fjxet)

    cjzxi = (fjxet * fjyze) - (fjyet * fjxze)
    cjzet = -(fjxxi * fjyze) + (fjyxi * fjxze)
    cjzze = (fjxxi * fjyet) - (fjyxi * fjxet)

    # Calculate partials:
    # this need only be done for l = 0,1,2,3 since, by symmetry,
    # (6,7,4,5) = - (0,1,2,3).
    b = np.ndarray([x.shape[0], 3, 8], x.dtype)
    b[:, 0, 0] = -cjxxi - cjxet - cjxze
    b[:, 0, 1] = cjxxi - cjxet - cjxze
    b[:, 0, 2] = cjxxi + cjxet - cjxze
    b[:, 0, 3] = -cjxxi + cjxet - cjxze
    b[:, 0, 4] = -b[:, 0, 2]
    b[:, 0, 5] = -b[:, 0, 3]
    b[:, 0, 6] = -b[:, 0, 0]
    b[:, 0, 7] = -b[:, 0, 1]

    b[:, 1, 0] = -cjyxi - cjyet - cjyze
    b[:, 1, 1] = cjyxi - cjyet - cjyze
    b[:, 1, 2] = cjyxi + cjyet - cjyze
    b[:, 1, 3] = -cjyxi + cjyet - cjyze
    b[:, 1, 4] = -b[:, 1, 2]
    b[:, 1, 5] = -b[:, 1, 3]
    b[:, 1, 6] = -b[:, 1, 0]
    b[:, 1, 7] = -b[:, 1, 1]

    b[:, 2, 0] = -cjzxi - cjzet - cjzze
    b[:, 2, 1] = cjzxi - cjzet - cjzze
    b[:, 2, 2] = cjzxi + cjzet - cjzze
    b[:, 2, 3] = -cjzxi + cjzet - cjzze
    b[:, 2, 4] = -b[:, 2, 2]
    b[:, 2, 5] = -b[:, 2, 3]
    b[:, 2, 6] = -b[:, 2, 0]
    b[:, 2, 7] = -b[:, 2, 1]

    # Calculate jacobian determinant (volume)
    volume = 8 * (fjxet * cjxet + fjyet * cjyet + fjzet * cjzet)

    return b, volume


def sum_elem_face_normal(
    normal_x0: realarr,
    normal_y0: realarr,
    normal_z0: realarr,
    normal_x1: realarr,
    normal_y1: realarr,
    normal_z1: realarr,
    normal_x2: realarr,
    normal_y2: realarr,
    normal_z2: realarr,
    normal_x3: realarr,
    normal_y3: realarr,
    normal_z3: realarr,
    x0: RealT,
    y0: RealT,
    z0: RealT,
    x1: RealT,
    y1: RealT,
    z1: RealT,
    x2: RealT,
    y2: RealT,
    z2: RealT,
    x3: RealT,
    y3: RealT,
    z3: RealT,
):
    bisect_x0 = 0.5 * (x3 + x2 - x1 - x0)
    bisect_y0 = 0.5 * (y3 + y2 - y1 - y0)
    bisect_z0 = 0.5 * (z3 + z2 - z1 - z0)
    bisect_x1 = 0.5 * (x2 + x1 - x3 - x0)
    bisect_y1 = 0.5 * (y2 + y1 - y3 - y0)
    bisect_z1 = 0.5 * (z2 + z1 - z3 - z0)
    area_x = 0.25 * (bisect_y0 * bisect_z1 - bisect_z0 * bisect_y1)
    area_y = 0.25 * (bisect_z0 * bisect_x1 - bisect_x0 * bisect_z1)
    area_z = 0.25 * (bisect_x0 * bisect_y1 - bisect_y0 * bisect_x1)

    normal_x0 += area_x
    normal_x1 += area_x
    normal_x2 += area_x
    normal_x3 += area_x
    normal_y0 += area_y
    normal_y1 += area_y
    normal_y2 += area_y
    normal_y3 += area_y
    normal_z0 += area_z
    normal_z1 += area_z
    normal_z2 += area_z
    normal_z3 += area_z


def _calc_elem_node_face(
    pfx: realarr,
    pfy: realarr,
    pfz: realarr,
    x: realarr,
    y: realarr,
    z: realarr,
    nodes: Tuple[int],
):
    a, b, c, d = nodes
    sum_elem_face_normal(
        pfx[:, a],
        pfy[:, a],
        pfz[:, a],
        pfx[:, b],
        pfy[:, b],
        pfz[:, b],
        pfx[:, c],
        pfy[:, c],
        pfz[:, c],
        pfx[:, d],
        pfy[:, d],
        pfz[:, d],
        x[:, a],
        y[:, a],
        z[:, a],
        x[:, b],
        y[:, b],
        z[:, b],
        x[:, c],
        y[:, c],
        z[:, c],
        x[:, d],
        y[:, d],
        z[:, d],
    )


def calc_elem_node_normals(pf: realarr, x: realarr, y: realarr, z: realarr):
    pf[:, :, :] = 0
    pfx = pf[:, 0, :]
    pfy = pf[:, 1, :]
    pfz = pf[:, 2, :]

    # Evaluate face one: nodes 0, 1, 2, 3
    _calc_elem_node_face(pfx, pfy, pfz, x, y, z, (0, 1, 2, 3))
    # Evaluate face two: nodes 0, 4, 5, 1
    _calc_elem_node_face(pfx, pfy, pfz, x, y, z, (0, 4, 5, 1))
    # Evaluate face three: nodes 1, 5, 6, 2
    _calc_elem_node_face(pfx, pfy, pfz, x, y, z, (1, 5, 6, 2))
    # Evaluate face four: nodes 2, 6, 7, 3
    _calc_elem_node_face(pfx, pfy, pfz, x, y, z, (2, 6, 7, 3))
    # Evaluate face five: nodes 3, 7, 4, 0
    _calc_elem_node_face(pfx, pfy, pfz, x, y, z, (3, 7, 4, 0))
    # Evaluate face six: nodes 4, 7, 6, 5
    _calc_elem_node_face(pfx, pfy, pfz, x, y, z, (4, 7, 6, 5))


def sum_elem_stresses_to_node_forces(
    B: realarr, stress_xx: realarr, stress_yy: realarr, stress_zz: realarr
):
    return (
        -stress_xx[:, None] * B[:, 0],
        -stress_yy[:, None] * B[:, 1],
        -stress_zz[:, None] * B[:, 2],
    )


def integrate_stress_for_elems(
    domain: Domain, sigxx: realarr, sigyy: realarr, sigzz: realarr
):
    x_local, y_local, z_local = collect_domain_nodes_to_elem_nodes(
        domain, domain.nodelist
    )
    B, determ = calc_elem_shape_function_derivatives(x_local, y_local, z_local)
    calc_elem_node_normals(B[:, :, :], x_local, y_local, z_local)
    fx_local, fy_local, fz_local = sum_elem_stresses_to_node_forces(
        B, sigxx, sigyy, sigzz
    )

    # Accumulate local force contributions to global array
    for i in range(8):
        nodelist = domain.nodelist[:, i]
        domain.fx[nodelist] += fx_local[:, i]
        domain.fy[nodelist] += fy_local[:, i]
        domain.fz[nodelist] += fz_local[:, i]

    return determ


def volu_der(
    x: realarr,
    y: realarr,
    z: realarr,
    dvdx: realarr,
    dvdy: realarr,
    dvdz: realarr,
    in_indices: Tuple[int],
    out_index: int,
):
    x0, x1, x2, x3, x4, x5 = np.split(x[:, in_indices], 6, axis=1)
    y0, y1, y2, y3, y4, y5 = np.split(y[:, in_indices], 6, axis=1)
    z0, z1, z2, z3, z4, z5 = np.split(z[:, in_indices], 6, axis=1)
    twelfth = 1 / 12
    o = out_index
    dvdx[:, o : o + 1] = twelfth * (
        (y1 + y2) * (z0 + z1)
        - (y0 + y1) * (z1 + z2)
        + (y0 + y4) * (z3 + z4)
        - (y3 + y4) * (z0 + z4)
        - (y2 + y5) * (z3 + z5)
        + (y3 + y5) * (z2 + z5)
    )
    dvdy[:, o : o + 1] = twelfth * (
        -(x1 + x2) * (z0 + z1)
        + (x0 + x1) * (z1 + z2)
        - (x0 + x4) * (z3 + z4)
        + (x3 + x4) * (z0 + z4)
        + (x2 + x5) * (z3 + z5)
        - (x3 + x5) * (z2 + z5)
    )
    dvdz[:, o : o + 1] = twelfth * (
        -(y1 + y2) * (x0 + x1)
        + (y0 + y1) * (x1 + x2)
        - (y0 + y4) * (x3 + x4)
        + (y3 + y4) * (x0 + x4)
        + (y2 + y5) * (x3 + x5)
        - (y3 + y5) * (x2 + x5)
    )


def calc_elem_volume_derivative(x: realarr, y: realarr, z: realarr):
    dvdx = np.empty_like(x)
    dvdy = np.empty_like(y)
    dvdz = np.empty_like(z)
    volu_der(x, y, z, dvdx, dvdy, dvdz, (1, 2, 3, 4, 5, 7), 0)
    volu_der(x, y, z, dvdx, dvdy, dvdz, (0, 1, 2, 7, 4, 6), 3)
    volu_der(x, y, z, dvdx, dvdy, dvdz, (3, 0, 1, 6, 7, 5), 2)
    volu_der(x, y, z, dvdx, dvdy, dvdz, (2, 3, 0, 5, 6, 4), 1)
    volu_der(x, y, z, dvdx, dvdy, dvdz, (7, 6, 5, 0, 3, 1), 4)
    volu_der(x, y, z, dvdx, dvdy, dvdz, (4, 7, 6, 1, 0, 2), 5)
    volu_der(x, y, z, dvdx, dvdy, dvdz, (5, 4, 7, 2, 1, 3), 6)
    volu_der(x, y, z, dvdx, dvdy, dvdz, (6, 5, 4, 3, 2, 0), 7)
    return dvdx, dvdy, dvdz


def calc_elem_fb_hourglass_force(
    xd: realarr, yd: realarr, zd: realarr, hourgam: realarr, coefficient: realarr
):
    hgfx = np.ndarray([xd.shape[0], 8], dtype=xd.dtype)
    hgfy = np.ndarray([xd.shape[0], 8], dtype=xd.dtype)
    hgfz = np.ndarray([xd.shape[0], 8], dtype=xd.dtype)

    hxx = np.einsum("eji,ej->ei", hourgam, xd)
    hgfx = coefficient[:, None] * np.einsum("eji,ei->ej", hourgam, hxx)
    hxx = np.einsum("eji,ej->ei", hourgam, yd)
    hgfy = coefficient[:, None] * np.einsum("eji,ei->ej", hourgam, hxx)
    hxx = np.einsum("eji,ej->ei", hourgam, zd)
    hgfz = coefficient[:, None] * np.einsum("eji,ei->ej", hourgam, hxx)

    return hgfx, hgfy, hgfz


def calc_fb_hourglass_force_for_elems(
    domain: Domain,
    determ: realarr,
    x8n: realarr,
    y8n: realarr,
    z8n: realarr,
    dvdx: realarr,
    dvdy: realarr,
    dvdz: realarr,
    hourg: float,
    numelem: int,
    numnode: int,
):
    """
    Calculates the Flanagan-Belytschko anti-hourglass force.
    """
    hourgam = np.ndarray([numelem, 8, 4], dtype=RealT)
    volinv = 1 / determ

    # Calculate hourglass modes
    for i in range(4):
        hourmodx = x8n @ gamma[i]
        hourmody = y8n @ gamma[i]
        hourmodz = z8n @ gamma[i]
        # Original code
        # hourmodx = (x8n[:, 0] * gamma[i, 0] + x8n[:, 1] * gamma[i, 1] +
        #             x8n[:, 2] * gamma[i, 2] + x8n[:, 3] * gamma[i, 3] +
        #             x8n[:, 4] * gamma[i, 4] + x8n[:, 5] * gamma[i, 5] +
        #             x8n[:, 6] * gamma[i, 6] + x8n[:, 7] * gamma[i, 7])
        # hourmody = (y8n[:, 0] * gamma[i, 0] + y8n[:, 1] * gamma[i, 1] +
        #             y8n[:, 2] * gamma[i, 2] + y8n[:, 3] * gamma[i, 3] +
        #             y8n[:, 4] * gamma[i, 4] + y8n[:, 5] * gamma[i, 5] +
        #             y8n[:, 6] * gamma[i, 6] + y8n[:, 7] * gamma[i, 7])
        # hourmodz = (z8n[:, 0] * gamma[i, 0] + z8n[:, 1] * gamma[i, 1] +
        #             z8n[:, 2] * gamma[i, 2] + z8n[:, 3] * gamma[i, 3] +
        #             z8n[:, 4] * gamma[i, 4] + z8n[:, 5] * gamma[i, 5] +
        #             z8n[:, 6] * gamma[i, 6] + z8n[:, 7] * gamma[i, 7])

        for j in range(8):
            hourgam[:, j, i] = gamma[i, j] - volinv * (
                dvdx[:, j] * hourmodx + dvdy[:, j] * hourmody + dvdz[:, j] * hourmodz
            )

    # Compute forces and store into force arrays
    volume13 = np.cbrt(determ)
    coefficient = -hourg * 0.01 * domain.ss * domain.elem_mass / volume13

    xd1 = domain.xd[domain.nodelist]
    yd1 = domain.yd[domain.nodelist]
    zd1 = domain.zd[domain.nodelist]

    hgfx, hgfy, hgfz = calc_elem_fb_hourglass_force(xd1, yd1, zd1, hourgam, coefficient)

    for i in range(8):
        nodelist = domain.nodelist[:, i]
        domain.fx[nodelist] += hgfx[:, i]
        domain.fy[nodelist] += hgfy[:, i]
        domain.fz[nodelist] += hgfz[:, i]


def calc_hourglass_control_for_elems(domain: Domain, determ: realarr, hgcoef: float):
    x1, y1, z1 = collect_domain_nodes_to_elem_nodes(domain, domain.nodelist)
    pfx, pfy, pfz = calc_elem_volume_derivative(x1, y1, z1)

    # Load into temporary storage for FB hourglass control
    x8n = np.ndarray([domain.numelem, 8])
    y8n = np.ndarray([domain.numelem, 8])
    z8n = np.ndarray([domain.numelem, 8])
    dvdx = np.ndarray([domain.numelem, 8])
    dvdy = np.ndarray([domain.numelem, 8])
    dvdz = np.ndarray([domain.numelem, 8])
    dvdx[:, :] = pfx
    dvdy[:, :] = pfy
    dvdz[:, :] = pfz
    x8n[:, :] = x1
    y8n[:, :] = y1
    z8n[:, :] = z1

    determ[:] = domain.volo * domain.v

    if np.any(domain.v <= 0):
        raise util.VolumeError

    if hgcoef > 0:
        calc_fb_hourglass_force_for_elems(
            domain,
            determ,
            x8n,
            y8n,
            z8n,
            dvdx,
            dvdy,
            dvdz,
            hgcoef,
            domain.numelem,
            domain.numnode,
        )


def calc_volume_force_for_elems(domain: Domain):
    if domain.numelem == 0:
        return

    # Sum contributions to stress tensor
    sigxx, sigyy, sigzz = init_stress_terms_for_elems(domain)

    # Produce nodal forces from material stresses
    determ = integrate_stress_for_elems(domain, sigxx, sigyy, sigzz)

    if np.any(determ <= 0):
        raise util.VolumeError

    calc_hourglass_control_for_elems(domain, determ, domain.hgcoef)


def calc_force_for_nodes(domain: Domain):
    domain.fx[:] = 0
    domain.fy[:] = 0
    domain.fz[:] = 0
    calc_volume_force_for_elems(domain)


def calc_acceleration_for_nodes(domain: Domain):
    domain.xdd[:] = domain.fx / domain.nodal_mass
    domain.ydd[:] = domain.fy / domain.nodal_mass
    domain.zdd[:] = domain.fz / domain.nodal_mass


def apply_acc_boundary_conditions_for_nodes(domain: Domain):
    if domain.symm_x.size > 0:
        domain.xdd[domain.symm_x] = 0
    if domain.symm_y.size > 0:
        domain.ydd[domain.symm_y] = 0
    if domain.symm_z.size > 0:
        domain.zdd[domain.symm_z] = 0


def calc_velocity_for_nodes(domain: Domain, dt: float, u_cut: float):
    xdtmp = domain.xd + domain.xdd * dt
    domain.xd[:] = np.where(np.abs(xdtmp) < u_cut, 0, xdtmp)
    ydtmp = domain.yd + domain.ydd * dt
    domain.yd[:] = np.where(np.abs(ydtmp) < u_cut, 0, ydtmp)
    zdtmp = domain.zd + domain.zdd * dt
    domain.zd[:] = np.where(np.abs(zdtmp) < u_cut, 0, zdtmp)


def calc_position_for_nodes(domain: Domain, dt: float):
    domain.x += domain.xd * dt
    domain.y += domain.yd * dt
    domain.z += domain.zd * dt


def lagrange_nodal(domain: Domain):
    """
    Compute nodal forces, acceleration, velocities, and positions w.r.t.
    boundary conditions and slide surface considerations.
    """
    calc_force_for_nodes(domain)
    calc_acceleration_for_nodes(domain)
    apply_acc_boundary_conditions_for_nodes(domain)
    calc_velocity_for_nodes(domain, domain.deltatime, domain.u_cut)
    calc_position_for_nodes(domain, domain.deltatime)


def _triple_product(x1, y1, z1, x2, y2, z2, x3, y3, z3):
    return (
        x1 * (y2 * z3 - z2 * y3) + x2 * (z1 * y3 - y1 * z3) + x3 * (y1 * z2 - z1 * y2)
    )


def calc_elem_volume(x: realarr, y: realarr, z: realarr):
    dx61 = x[:, 6] - x[:, 1]
    dy61 = y[:, 6] - y[:, 1]
    dz61 = z[:, 6] - z[:, 1]

    dx70 = x[:, 7] - x[:, 0]
    dy70 = y[:, 7] - y[:, 0]
    dz70 = z[:, 7] - z[:, 0]

    dx63 = x[:, 6] - x[:, 3]
    dy63 = y[:, 6] - y[:, 3]
    dz63 = z[:, 6] - z[:, 3]

    dx20 = x[:, 2] - x[:, 0]
    dy20 = y[:, 2] - y[:, 0]
    dz20 = z[:, 2] - z[:, 0]

    dx50 = x[:, 5] - x[:, 0]
    dy50 = y[:, 5] - y[:, 0]
    dz50 = z[:, 5] - z[:, 0]

    dx64 = x[:, 6] - x[:, 4]
    dy64 = y[:, 6] - y[:, 4]
    dz64 = z[:, 6] - z[:, 4]

    dx31 = x[:, 3] - x[:, 1]
    dy31 = y[:, 3] - y[:, 1]
    dz31 = z[:, 3] - z[:, 1]

    dx72 = x[:, 7] - x[:, 2]
    dy72 = y[:, 7] - y[:, 2]
    dz72 = z[:, 7] - z[:, 2]

    dx43 = x[:, 4] - x[:, 3]
    dy43 = y[:, 4] - y[:, 3]
    dz43 = z[:, 4] - z[:, 3]

    dx57 = x[:, 5] - x[:, 7]
    dy57 = y[:, 5] - y[:, 7]
    dz57 = z[:, 5] - z[:, 7]

    dx14 = x[:, 1] - x[:, 4]
    dy14 = y[:, 1] - y[:, 4]
    dz14 = z[:, 1] - z[:, 4]

    dx25 = x[:, 2] - x[:, 5]
    dy25 = y[:, 2] - y[:, 5]
    dz25 = z[:, 2] - z[:, 5]

    volume = (
        _triple_product(
            dx31 + dx72, dx63, dx20, dy31 + dy72, dy63, dy20, dz31 + dz72, dz63, dz20
        )
        + _triple_product(
            dx43 + dx57, dx64, dx70, dy43 + dy57, dy64, dy70, dz43 + dz57, dz64, dz70
        )
        + _triple_product(
            dx14 + dx25, dx61, dx50, dy14 + dy25, dy61, dy50, dz14 + dz25, dz61, dz50
        )
    )

    volume /= 12.0
    return volume


def area_face(x0, x1, x2, x3, y0, y1, y2, y3, z0, z1, z2, z3):
    fx = (x2 - x0) - (x3 - x1)
    fy = (y2 - y0) - (y3 - y1)
    fz = (z2 - z0) - (z3 - z1)
    gx = (x2 - x0) + (x3 - x1)
    gy = (y2 - y0) + (y3 - y1)
    gz = (z2 - z0) + (z3 - z1)
    area = (fx * fx + fy * fy + fz * fz) * (gx * gx + gy * gy + gz * gz) - (
        fx * gx + fy * gy + fz * gz
    ) * (fx * gx + fy * gy + fz * gz)
    return area


def calc_elem_characteristic_length(
    x: realarr, y: realarr, z: realarr, volume: realarr
):
    char_length = np.zeros_like(volume)

    a = area_face(
        x[:, 0],
        x[:, 1],
        x[:, 2],
        x[:, 3],
        y[:, 0],
        y[:, 1],
        y[:, 2],
        y[:, 3],
        z[:, 0],
        z[:, 1],
        z[:, 2],
        z[:, 3],
    )
    char_length = np.maximum(a, char_length)

    a = area_face(
        x[:, 4],
        x[:, 5],
        x[:, 6],
        x[:, 7],
        y[:, 4],
        y[:, 5],
        y[:, 6],
        y[:, 7],
        z[:, 4],
        z[:, 5],
        z[:, 6],
        z[:, 7],
    )
    char_length = np.maximum(a, char_length)

    a = area_face(
        x[:, 0],
        x[:, 1],
        x[:, 5],
        x[:, 4],
        y[:, 0],
        y[:, 1],
        y[:, 5],
        y[:, 4],
        z[:, 0],
        z[:, 1],
        z[:, 5],
        z[:, 4],
    )
    char_length = np.maximum(a, char_length)

    a = area_face(
        x[:, 1],
        x[:, 2],
        x[:, 6],
        x[:, 5],
        y[:, 1],
        y[:, 2],
        y[:, 6],
        y[:, 5],
        z[:, 1],
        z[:, 2],
        z[:, 6],
        z[:, 5],
    )
    char_length = np.maximum(a, char_length)

    a = area_face(
        x[:, 2],
        x[:, 3],
        x[:, 7],
        x[:, 6],
        y[:, 2],
        y[:, 3],
        y[:, 7],
        y[:, 6],
        z[:, 2],
        z[:, 3],
        z[:, 7],
        z[:, 6],
    )
    char_length = np.maximum(a, char_length)

    a = area_face(
        x[:, 3],
        x[:, 0],
        x[:, 4],
        x[:, 7],
        y[:, 3],
        y[:, 0],
        y[:, 4],
        y[:, 7],
        z[:, 3],
        z[:, 0],
        z[:, 4],
        z[:, 7],
    )
    char_length = np.maximum(a, char_length)

    char_length = 4.0 * volume / np.sqrt(char_length)

    return char_length


def calc_elem_velocity_gradient(
    xvel: realarr, yvel: realarr, zvel: realarr, b: realarr, detJ: realarr
):
    inv_detJ = 1.0 / detJ
    pfx = b[:, 0]
    pfy = b[:, 1]
    pfz = b[:, 2]

    d = np.ndarray([xvel.shape[0], 6], xvel.dtype)
    d[:, 0] = inv_detJ * (
        pfx[:, 0] * (xvel[:, 0] - xvel[:, 6])
        + pfx[:, 1] * (xvel[:, 1] - xvel[:, 7])
        + pfx[:, 2] * (xvel[:, 2] - xvel[:, 4])
        + pfx[:, 3] * (xvel[:, 3] - xvel[:, 5])
    )

    d[:, 1] = inv_detJ * (
        pfy[:, 0] * (yvel[:, 0] - yvel[:, 6])
        + pfy[:, 1] * (yvel[:, 1] - yvel[:, 7])
        + pfy[:, 2] * (yvel[:, 2] - yvel[:, 4])
        + pfy[:, 3] * (yvel[:, 3] - yvel[:, 5])
    )

    d[:, 2] = inv_detJ * (
        pfz[:, 0] * (zvel[:, 0] - zvel[:, 6])
        + pfz[:, 1] * (zvel[:, 1] - zvel[:, 7])
        + pfz[:, 2] * (zvel[:, 2] - zvel[:, 4])
        + pfz[:, 3] * (zvel[:, 3] - zvel[:, 5])
    )

    dyddx = inv_detJ * (
        pfx[:, 0] * (yvel[:, 0] - yvel[:, 6])
        + pfx[:, 1] * (yvel[:, 1] - yvel[:, 7])
        + pfx[:, 2] * (yvel[:, 2] - yvel[:, 4])
        + pfx[:, 3] * (yvel[:, 3] - yvel[:, 5])
    )

    dxddy = inv_detJ * (
        pfy[:, 0] * (xvel[:, 0] - xvel[:, 6])
        + pfy[:, 1] * (xvel[:, 1] - xvel[:, 7])
        + pfy[:, 2] * (xvel[:, 2] - xvel[:, 4])
        + pfy[:, 3] * (xvel[:, 3] - xvel[:, 5])
    )

    dzddx = inv_detJ * (
        pfx[:, 0] * (zvel[:, 0] - zvel[:, 6])
        + pfx[:, 1] * (zvel[:, 1] - zvel[:, 7])
        + pfx[:, 2] * (zvel[:, 2] - zvel[:, 4])
        + pfx[:, 3] * (zvel[:, 3] - zvel[:, 5])
    )

    dxddz = inv_detJ * (
        pfz[:, 0] * (xvel[:, 0] - xvel[:, 6])
        + pfz[:, 1] * (xvel[:, 1] - xvel[:, 7])
        + pfz[:, 2] * (xvel[:, 2] - xvel[:, 4])
        + pfz[:, 3] * (xvel[:, 3] - xvel[:, 5])
    )

    dzddy = inv_detJ * (
        pfy[:, 0] * (zvel[:, 0] - zvel[:, 6])
        + pfy[:, 1] * (zvel[:, 1] - zvel[:, 7])
        + pfy[:, 2] * (zvel[:, 2] - zvel[:, 4])
        + pfy[:, 3] * (zvel[:, 3] - zvel[:, 5])
    )

    dyddz = inv_detJ * (
        pfz[:, 0] * (yvel[:, 0] - yvel[:, 6])
        + pfz[:, 1] * (yvel[:, 1] - yvel[:, 7])
        + pfz[:, 2] * (yvel[:, 2] - yvel[:, 4])
        + pfz[:, 3] * (yvel[:, 3] - yvel[:, 5])
    )
    d[:, 5] = 0.5 * (dxddy + dyddx)
    d[:, 4] = 0.5 * (dxddz + dzddx)
    d[:, 3] = 0.5 * (dzddy + dyddz)
    return d


def calc_kinematics_for_elems(domain: Domain):
    # Get nodal coordinates from global arrays and copy into local arrays
    x_local, y_local, z_local = collect_domain_nodes_to_elem_nodes(
        domain, domain.nodelist
    )

    # Volume calculations
    volume = calc_elem_volume(x_local, y_local, z_local)
    relative_volume = volume / domain.volo
    domain.vnew[:] = relative_volume
    domain.delv[:] = relative_volume - domain.v

    # Set characteristic length
    domain.arealg[:] = calc_elem_characteristic_length(
        x_local, y_local, z_local, volume
    )

    # Get nodal velocities from global arrays and copy into local arrays
    xd_local = domain.xd[domain.nodelist]
    yd_local = domain.yd[domain.nodelist]
    zd_local = domain.zd[domain.nodelist]
    dt2 = 0.5 * domain.deltatime

    x_local -= dt2 * xd_local
    y_local -= dt2 * yd_local
    z_local -= dt2 * zd_local

    B, det_J = calc_elem_shape_function_derivatives(x_local, y_local, z_local)
    D = calc_elem_velocity_gradient(xd_local, yd_local, zd_local, B, det_J)

    domain.dxx = D[:, 0]
    domain.dyy = D[:, 1]
    domain.dzz = D[:, 2]


def calc_lagrange_elements(domain: Domain):
    if domain.numelem == 0:
        return

    calc_kinematics_for_elems(domain)

    # Calculate strain rate and apply as constraint (only done in FB element)
    domain.vdov[:] = domain.dxx + domain.dyy + domain.dzz

    # Make the rate of deformation tensor deviatoric
    vdovthird = domain.vdov / 3
    domain.dxx -= vdovthird
    domain.dyy -= vdovthird
    domain.dzz -= vdovthird

    if np.any(domain.vnew <= 0):
        raise util.VolumeError


def calc_monotonic_q_gradients_for_elems(domain: Domain):
    n = domain.nodelist
    shp = (n.shape[1], n.shape[0])

    x = np.ndarray(shp, domain.x.dtype)
    y = np.ndarray(shp, domain.y.dtype)
    z = np.ndarray(shp, domain.z.dtype)
    xv = np.ndarray(shp, domain.xd.dtype)
    yv = np.ndarray(shp, domain.yd.dtype)
    zv = np.ndarray(shp, domain.zd.dtype)
    for i in range(n.shape[1]):
        x[i] = domain.x[n[:, i]]
        y[i] = domain.y[n[:, i]]
        z[i] = domain.z[n[:, i]]
        xv[i] = domain.xd[n[:, i]]
        yv[i] = domain.yd[n[:, i]]
        zv[i] = domain.zd[n[:, i]]

    vol = domain.volo * domain.vnew
    norm = 1 / (vol + ptiny)

    dxj = -0.25 * ((x[0] + x[1] + x[5] + x[4]) - (x[3] + x[2] + x[6] + x[7]))
    dyj = -0.25 * ((y[0] + y[1] + y[5] + y[4]) - (y[3] + y[2] + y[6] + y[7]))
    dzj = -0.25 * ((z[0] + z[1] + z[5] + z[4]) - (z[3] + z[2] + z[6] + z[7]))

    dxi = 0.25 * ((x[1] + x[2] + x[6] + x[5]) - (x[0] + x[3] + x[7] + x[4]))
    dyi = 0.25 * ((y[1] + y[2] + y[6] + y[5]) - (y[0] + y[3] + y[7] + y[4]))
    dzi = 0.25 * ((z[1] + z[2] + z[6] + z[5]) - (z[0] + z[3] + z[7] + z[4]))

    dxk = 0.25 * ((x[4] + x[5] + x[6] + x[7]) - (x[0] + x[1] + x[2] + x[3]))
    dyk = 0.25 * ((y[4] + y[5] + y[6] + y[7]) - (y[0] + y[1] + y[2] + y[3]))
    dzk = 0.25 * ((z[4] + z[5] + z[6] + z[7]) - (z[0] + z[1] + z[2] + z[3]))

    ax = dyi * dzj - dzi * dyj
    ay = dzi * dxj - dxi * dzj
    az = dxi * dyj - dyi * dxj

    domain.delx_zeta = vol / np.sqrt(ax * ax + ay * ay + az * az + ptiny)

    ax *= norm
    ay *= norm
    az *= norm

    dxv = 0.25 * ((xv[4] + xv[5] + xv[6] + xv[7]) - (xv[0] + xv[1] + xv[2] + xv[3]))
    dyv = 0.25 * ((yv[4] + yv[5] + yv[6] + yv[7]) - (yv[0] + yv[1] + yv[2] + yv[3]))
    dzv = 0.25 * ((zv[4] + zv[5] + zv[6] + zv[7]) - (zv[0] + zv[1] + zv[2] + zv[3]))

    domain.delv_zeta = ax * dxv + ay * dyv + az * dzv

    # find delxi and delvi ( j cross k )

    ax = dyj * dzk - dzj * dyk
    ay = dzj * dxk - dxj * dzk
    az = dxj * dyk - dyj * dxk

    domain.delx_xi = vol / np.sqrt(ax * ax + ay * ay + az * az + ptiny)

    ax *= norm
    ay *= norm
    az *= norm

    dxv = 0.25 * ((xv[1] + xv[2] + xv[6] + xv[5]) - (xv[0] + xv[3] + xv[7] + xv[4]))
    dyv = 0.25 * ((yv[1] + yv[2] + yv[6] + yv[5]) - (yv[0] + yv[3] + yv[7] + yv[4]))
    dzv = 0.25 * ((zv[1] + zv[2] + zv[6] + zv[5]) - (zv[0] + zv[3] + zv[7] + zv[4]))

    domain.delv_xi = ax * dxv + ay * dyv + az * dzv

    # find delxj and delvj ( k cross i )

    ax = dyk * dzi - dzk * dyi
    ay = dzk * dxi - dxk * dzi
    az = dxk * dyi - dyk * dxi

    domain.delx_eta = vol / np.sqrt(ax * ax + ay * ay + az * az + ptiny)

    ax *= norm
    ay *= norm
    az *= norm

    dxv = -0.25 * ((xv[0] + xv[1] + xv[5] + xv[4]) - (xv[3] + xv[2] + xv[6] + xv[7]))
    dyv = -0.25 * ((yv[0] + yv[1] + yv[5] + yv[4]) - (yv[3] + yv[2] + yv[6] + yv[7]))
    dzv = -0.25 * ((zv[0] + zv[1] + zv[5] + zv[4]) - (zv[3] + zv[2] + zv[6] + zv[7]))

    domain.delv_eta = ax * dxv + ay * dyv + az * dzv


def _calc_monotonic_q_region_bc(
    domain: Domain,
    bc: Dict[str, Dict[str, int]],
    bc_mask: intarr,
    ielem: intarr,
    delv: realarr,
    lm: intarr,
    lp: intarr,
) -> realarr:
    """
    Helper function that computes two boundary condition faces, used in
    ``calc_monotonic_q_region_for_elems``.
    """
    delv_ielem = delv[ielem]
    norm = 1.0 / (delv_ielem + ptiny)

    # masked == *_FREE uses default value
    masked = bc_mask & bc["M"]["mask"]
    delvm = np.select(
        [(masked == bc["M"]["COMM"]) | (masked == 0), masked == bc["M"]["SYMM"]],
        [delv[lm[ielem]], delv_ielem],
        default=0,
    )

    # masked == *_FREE uses default value
    masked = bc_mask & bc["P"]["mask"]
    delvp = np.select(
        [(masked == bc["P"]["COMM"]) | (masked == 0), masked == bc["P"]["SYMM"]],
        [delv[lp[ielem]], delv_ielem],
        default=0,
    )

    delvm *= norm
    delvp *= norm
    phi = 0.5 * (delvm + delvp)

    delvm *= domain.monoq_limiter_mult
    delvp *= domain.monoq_limiter_mult

    phi = np.minimum(phi, delvm)
    phi = np.minimum(phi, delvp)
    phi = np.clip(phi, 0, domain.monoq_max_slope)
    return phi


def calc_monotonic_q_region_for_elems(domain: Domain, r: int):
    ielem = domain.reg_elem_list[r]
    bc_mask = domain.elem_bc[ielem]
    phixi = _calc_monotonic_q_region_bc(
        domain, XI, bc_mask, ielem, domain.delv_xi, domain.lxim, domain.lxip
    )
    phieta = _calc_monotonic_q_region_bc(
        domain, ETA, bc_mask, ielem, domain.delv_eta, domain.letam, domain.letap
    )
    phizeta = _calc_monotonic_q_region_bc(
        domain, ZETA, bc_mask, ielem, domain.delv_zeta, domain.lzetam, domain.lzetap
    )

    # Remove length scale
    delvx_xi = np.minimum(0, domain.delv_xi[ielem] * domain.delx_xi[ielem])
    delvx_eta = np.minimum(0, domain.delv_eta[ielem] * domain.delx_eta[ielem])
    delvx_zeta = np.minimum(0, domain.delv_zeta[ielem] * domain.delx_zeta[ielem])
    rho = domain.elem_mass[ielem] / (domain.volo[ielem] * domain.vnew[ielem])
    qlin = (
        -domain.qlc_monoq
        * rho
        * (
            delvx_xi * (1 - phixi)
            + delvx_eta * (1 - phieta)
            + delvx_zeta * (1 - phizeta)
        )
    )
    qquad = (
        domain.qqc_monoq
        * rho
        * (
            delvx_xi**2 * (1 - phixi**2)
            + delvx_eta**2 * (1 - phieta**2)
            + delvx_zeta**2 * (1 - phizeta**2)
        )
    )

    domain.qq[ielem] = np.where(domain.vdov[ielem] > 0, 0, qquad)
    domain.ql[ielem] = np.where(domain.vdov[ielem] > 0, 0, qlin)


def calc_monotonic_q_for_elems(domain: Domain):
    # Calculate monotonic q for all regions
    for r in range(domain.numregions):
        if domain.reg_elem_size[r] > 0:
            calc_monotonic_q_region_for_elems(domain, r)


def calc_q_for_elems(domain: Domain):
    """
    MONOTONIC Q option
    """
    calc_monotonic_q_gradients_for_elems(domain)
    calc_monotonic_q_for_elems(domain)

    if np.any(domain.q > domain.qstop):
        raise util.QStopError


def calc_pressure_for_elems(
    p_new: realarr,
    bvc: realarr,
    pbvc: realarr,
    e_old: realarr,
    compression: realarr,
    vnewc_elem: realarr,
    pmin: float,
    p_cut: float,
    eos_vmax: float,
):
    c1s = 2 / 3
    bvc[:] = c1s * (compression + 1)
    pbvc[:] = c1s

    p_new[:] = bvc * e_old
    p_new[:] = np.where(np.abs(p_new) < p_cut, 0, p_new)
    # This condition may never happen
    p_new[:] = np.where(vnewc_elem >= eos_vmax, 0, p_new)
    p_new[:] = np.maximum(p_new, pmin)


def calc_energy_for_elems(
    p_new: realarr,
    e_new: realarr,
    q_new: realarr,
    bvc: realarr,
    pbvc: realarr,
    p_old: realarr,
    e_old: realarr,
    q_old: realarr,
    compression: realarr,
    comp_half_step: realarr,
    vnewc_elem: realarr,
    work: realarr,
    delvc: realarr,
    pmin: float,
    p_cut: float,
    e_cut: float,
    q_cut: float,
    emin: float,
    qq_old: realarr,
    ql_old: realarr,
    rho0: float,
    eos_vmax: float,
    region_elems: intarr,
):
    p_half_step = np.empty([region_elems.shape[0]], comp_half_step.dtype)

    e_new[:] = np.maximum(e_old - 0.5 * delvc * (p_old + q_old) + 0.5 * work, emin)

    # Modifies bvc, pbvc, p_half_step
    calc_pressure_for_elems(
        p_half_step, bvc, pbvc, e_new, comp_half_step, vnewc_elem, pmin, p_cut, eos_vmax
    )

    vhalf = 1 / (1 + comp_half_step)
    ssc = (pbvc * e_new + vhalf * vhalf * bvc * p_half_step) / rho0
    ssc[:] = np.where(ssc <= 0.1111111e-36, 0.3333333e-18, np.sqrt(ssc))
    q_new[:] = np.where(delvc > 0, 0, ssc * ql_old + qq_old)

    e_new += 0.5 * delvc * (3 * (p_old + q_old) - 4 * (p_half_step + q_new))

    e_new += 0.5 * work
    e_new[:] = np.where(np.abs(e_new) < e_cut, 0, e_new)
    e_new[:] = np.maximum(e_new, emin)

    # Modifies bvc, pbvc, p_new
    calc_pressure_for_elems(
        p_new, bvc, pbvc, e_new, compression, vnewc_elem, pmin, p_cut, eos_vmax
    )

    sixth = 1 / 6
    ssc = (pbvc * e_new + vnewc_elem * vnewc_elem * bvc * p_new) / rho0
    ssc[:] = np.where(ssc <= 0.1111111e-36, 0.3333333e-18, np.sqrt(ssc))
    q_tilde = np.where(delvc > 0, 0, ssc * ql_old + qq_old)

    e_new -= (
        (7 * (p_old + q_old) - 8 * (p_half_step + q_new) + (p_new + q_tilde))
        * delvc
        * sixth
    )
    e_new[:] = np.where(np.abs(e_new) < e_cut, 0, e_new)
    e_new[:] = np.maximum(e_new, emin)

    # Modifies bvc, pbvc, p_new
    calc_pressure_for_elems(
        p_new, bvc, pbvc, e_new, compression, vnewc_elem, pmin, p_cut, eos_vmax
    )

    ssc = (pbvc * e_new + vnewc_elem * vnewc_elem * bvc * p_new) / rho0
    ssc[:] = np.where(ssc <= 0.1111111e-36, 0.3333333e-18, np.sqrt(ssc))
    expr = ssc * ql_old + qq_old

    q_new[:] = np.where(delvc <= 0, np.where(expr < q_cut, 0, expr), q_new)


def calc_sound_speed_for_elems(
    domain: Domain,
    vnewc_elem: realarr,
    rho0: float,
    enewc: realarr,
    pnewc: realarr,
    pbvc: realarr,
    bvc: realarr,
    region_elems: intarr,
):
    sstmp = (pbvc * enewc + vnewc_elem * vnewc_elem * bvc * pnewc) / rho0
    domain.ss[region_elems] = np.where(
        sstmp <= 0.1111111e-36, 0.3333333e-18, np.sqrt(sstmp)
    )


def eval_eos_for_elems(domain: Domain, vnewc: realarr, region_elems: intarr, rep: int):
    bvc = np.empty([region_elems.shape[0]], RealT)
    pbvc = np.empty([region_elems.shape[0]], RealT)
    p_new = np.empty([region_elems.shape[0]], RealT)
    e_new = np.empty([region_elems.shape[0]], RealT)
    q_new = np.empty([region_elems.shape[0]], RealT)

    vnewc_elem = vnewc[region_elems]

    # NOTE from original implementation:
    # Loop to add load imbalance based on region number
    for _ in range(rep):
        # These temporaries will be of different size for each call,
        # due to different sized region element lists
        e_old = np.copy(domain.e[region_elems])
        delvc = np.copy(domain.delv[region_elems])
        p_old = np.copy(domain.p[region_elems])
        q_old = np.copy(domain.q[region_elems])
        qq_old = np.copy(domain.qq[region_elems])
        ql_old = np.copy(domain.ql[region_elems])
        compression = 1 / vnewc_elem - 1
        vchalf = vnewc_elem - delvc * 0.5
        comp_half_step = 1 / vchalf - 1

        # NOTE: The following are impossible due to the clipping in
        # apply_material_properties_for_elems
        if domain.eos_vmin != 0:
            comp_half_step[:] = np.where(
                vnewc_elem <= domain.eos_vmin, compression, comp_half_step
            )
        if domain.eos_vmax != 0:
            p_old[:] = np.where(vnewc_elem >= domain.eos_vmax, 0, p_old)
            compression[:] = np.where(vnewc_elem >= domain.eos_vmax, 0, compression)
            comp_half_step[:] = np.where(
                vnewc_elem >= domain.eos_vmax, 0, comp_half_step
            )

        work = np.zeros([region_elems.shape[0]], RealT)

        calc_energy_for_elems(
            p_new,
            e_new,
            q_new,
            bvc,
            pbvc,
            p_old,
            e_old,
            q_old,
            compression,
            comp_half_step,
            vnewc_elem,
            work,
            delvc,
            domain.pmin,
            domain.p_cut,
            domain.e_cut,
            domain.q_cut,
            domain.emin,
            qq_old,
            ql_old,
            domain.refdens,
            domain.eos_vmax,
            region_elems,
        )
    # End of load imbalance loop

    domain.p[region_elems] = p_new
    domain.e[region_elems] = e_new
    domain.q[region_elems] = q_new

    calc_sound_speed_for_elems(
        domain, vnewc_elem, domain.refdens, e_new, p_new, pbvc, bvc, region_elems
    )


def apply_material_properties_for_elems(domain: Domain):
    if domain.numelem == 0:
        return

    # Bound the updated relative volumes
    lower_bound = domain.eos_vmin if domain.eos_vmin != 0 else -np.inf
    upper_bound = domain.eos_vmax if domain.eos_vmax != 0 else np.inf
    vnewc = np.clip(domain.vnew, lower_bound, upper_bound)

    # NOTE from original implementation:
    # This check may not make perfect sense in LULESH, but
    # it's representative of something in the full code -
    # just leave it in, please
    vc = np.clip(domain.v, lower_bound, upper_bound)
    if np.any(vc <= 0):
        raise util.VolumeError

    for r in range(domain.numregions):
        # Get region elements
        elems = domain.reg_elem_list[r, : domain.reg_elem_size[r]]

        # Determine load imbalance for this region
        # Round down the number with lowest cost
        if r < domain.numregions // 2:
            rep = 1
        elif r < (domain.numregions - (domain.numregions + 15) // 20):
            # NOTE from original implementation:
            # You don't get an expensive region unless you at least have 5 regions
            rep = 1 + domain.cost
        else:  # Very expensive regions
            rep = 10 * (1 + domain.cost)

        eval_eos_for_elems(domain, vnewc, elems, rep)


def update_volumes_for_elems(domain: Domain, v_cut: float):
    domain.v[:] = np.where(np.abs(domain.vnew - 1) < v_cut, 1.0, domain.vnew)


def lagrange_elements(domain: Domain):
    """
    Compute element quantities and update material properties.
    """
    calc_lagrange_elements(domain)
    calc_q_for_elems(domain)
    apply_material_properties_for_elems(domain)
    update_volumes_for_elems(domain, domain.v_cut)


def calc_courant_constraint_for_elems(
    domain: Domain, elems: intarr, qqc: float, dtcourant: float
) -> float:
    """
    Compute time constraints and potentially update time increment.

    :return: New courant time increment.
    """
    qqc2 = 64 * (qqc**2)

    dtf = domain.ss[elems] ** 2

    vdov = domain.vdov[elems]
    dtf = np.where(vdov < 0, dtf + qqc2 * (domain.arealg[elems] ** 2) * (vdov**2), dtf)

    dtf = domain.arealg[elems] / np.sqrt(dtf)

    dtmin = np.min(np.where(vdov != 0, dtf, np.inf), initial=np.inf)
    if dtmin < dtcourant:
        return dtmin

    return dtcourant


def calc_hydro_constraint_for_elems(
    domain: Domain, elems: intarr, dvovmax: float, dthydro: float
) -> float:
    """
    Compute hydro constraints and potentially update time increment.

    :return: New hydro time increment.
    """
    vdov = domain.vdov[elems]
    dthydro_min = np.min(
        np.where(vdov != 0, dvovmax / (np.abs(vdov) + 1e-20), np.inf), initial=np.inf
    )
    if dthydro_min < dthydro:
        return dthydro_min
    return dthydro


def calc_time_constraints_for_elems(domain: Domain):
    """
    Evaluate region constraints for elements.
    """
    # Initialize conditions to large values
    domain.dthydro = domain.dtcourant = 1e20

    for r in range(domain.numregions):
        # Get region elements
        elems = domain.reg_elem_list[r, : domain.reg_elem_size[r]]

        # Evaluate time constraint
        domain.dtcourant = calc_courant_constraint_for_elems(
            domain, elems, domain.qqc, domain.dtcourant
        )

        # Evaluate hydro constraint
        domain.dthydro = calc_hydro_constraint_for_elems(
            domain, elems, domain.dvovmax, domain.dthydro
        )


def lagrange_leapfrog(domain: Domain):
    """
    Leapfrog integration.
    """
    # Calculate nodal forces/acceleration/velocities/positions
    # with boundary conditions and slide surface considerations
    lagrange_nodal(domain)

    # Calculate element quantities and update material states
    lagrange_elements(domain)

    # Evaluate time and hydro constraints
    calc_time_constraints_for_elems(domain)
