import numpy as np


def rphi_to_xy(r, phi):
    return r * np.cos(phi), r * np.sin(phi)


def canonical_to_global(scan_r, scan_phi, dx, dy):
    tmp_y = scan_r + dy
    tmp_phi = np.arctan2(dx, tmp_y)
    dets_phi = tmp_phi + scan_phi
    dets_r = tmp_y / np.cos(tmp_phi)
    return dets_r, dets_phi


def canonical_to_global_xy(scan_r, scan_phi, dx, dy):
    dets_r, dets_phi = canonical_to_global(scan_r, scan_phi, dx, dy)
    return rphi_to_xy(dets_r, dets_phi)


def nms_predicted_center(scan_grid, phi_grid, pred_cls, pred_reg, min_dist=0.5):
    pred_r, pred_phi = canonical_to_global(
        scan_grid,
        phi_grid,
        pred_reg[:, 0],
        pred_reg[:, 1],
    )
    pred_xs, pred_ys = rphi_to_xy(pred_r, pred_phi)

    sort_inds = np.argsort(pred_cls)[::-1]
    pred_xs = pred_xs[sort_inds]
    pred_ys = pred_ys[sort_inds]
    pred_cls = pred_cls[sort_inds]

    num_pts = len(scan_grid)
    xdiff = pred_xs.reshape(num_pts, 1) - pred_xs.reshape(1, num_pts)
    ydiff = pred_ys.reshape(num_pts, 1) - pred_ys.reshape(1, num_pts)
    p_dist = np.sqrt(np.square(xdiff) + np.square(ydiff))

    keep = np.ones(num_pts, dtype=np.bool_)
    instance_mask = np.zeros(num_pts, dtype=np.int32)
    instance_id = 1

    for i in range(num_pts):
        if not keep[i]:
            continue
        dup_inds = p_dist[i] < min_dist
        keep[dup_inds] = False
        keep[i] = True
        instance_mask[sort_inds[dup_inds]] = instance_id
        instance_id += 1

    det_xys = np.stack((pred_xs, pred_ys), axis=1)[keep]
    det_cls = pred_cls[keep]
    return det_xys.astype(np.float32), det_cls.astype(np.float32), instance_mask