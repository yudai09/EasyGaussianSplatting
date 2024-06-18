import numpy as np

def remove_invalid_gs(gs):
    pws = gs['pw']
    x = pws[:, 0]
    y = pws[:, 1]
    z = pws[:, 2]
    valid_inds = np.logical_not(np.isnan(x + y + z))
    keys = ['pw', 'scale', 'sh', 'rot', 'alpha']
    new_gs = {}
    for key in keys:
        new_gs[key] = gs[key][valid_inds]
    return new_gs


def remove_outof_fov(gs, fov, tcw, Rcw):
    pws = gs['pw']
    x = pws[:, 0]
    y = pws[:, 1]
    z = pws[:, 2]

    r = np.sqrt(x * x + y * y) + 1e-3
    # insident angle
    theta = np.arctan2(r, z)

    # valid_inds = np.logical_and(theta < fov, theta > 0)

    # temporary solution
    # FIXME: 
    pc = (Rcw @ pws.T).T + tcw
    valid_inds = pc[:, 2] > 0.2

    keys = ['pw', 'scale', 'sh', 'rot', 'alpha']
    new_gs = {}
    for key in keys:
        new_gs[key] = gs[key][valid_inds]
    return new_gs

