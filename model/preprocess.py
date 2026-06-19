import math
import numpy as np

if "clip" in dir(np.core.umath):
    _clip = np.core.umath.clip
else:
    _clip = np.clip


def scans_to_cutout(
    scans,
    scan_phi,
    stride=1,
    centered=True,
    fixed=True,
    window_width=1.0,
    window_depth=0.5,
    num_cutout_pts=56,
    padding_val=29.99,
    area_mode=True,
):
    num_scans, num_pts = scans.shape

    dists = scans[:, ::stride] if fixed else np.tile(scans[-1, ::stride], num_scans).reshape(num_scans, -1)
    half_alpha = np.arctan(0.5 * window_width / np.maximum(dists, 1e-2))

    delta_alpha = 2.0 * half_alpha / (num_cutout_pts - 1)
    ang_ct = (
        scan_phi[::stride]
        - half_alpha
        + np.arange(num_cutout_pts, dtype=np.float32).reshape(num_cutout_pts, 1, 1) * delta_alpha
    )
    ang_ct = (ang_ct + np.pi) % (2.0 * np.pi) - np.pi
    inds_ct = (ang_ct - scan_phi[0]) / (scan_phi[1] - scan_phi[0])
    outbound_mask = np.logical_or(inds_ct < 0, inds_ct > num_pts - 1)

    inds_ct_low = _clip(np.floor(inds_ct), 0, num_pts - 1).astype(np.int32)
    inds_ct_high = _clip(inds_ct_low + 1, 0, num_pts - 1).astype(np.int32)
    inds_ct_ratio = _clip(inds_ct - inds_ct_low, 0.0, 1.0)

    inds_offset = np.arange(num_scans).reshape(1, num_scans, 1) * num_pts
    ct_low = np.take(scans, inds_ct_low + inds_offset)
    ct_high = np.take(scans, inds_ct_high + inds_offset)
    ct = ct_low + inds_ct_ratio * (ct_high - ct_low)

    if area_mode:
        num_pts_in_window = inds_ct[-1] - inds_ct[0]
        area_mask = num_pts_in_window > num_cutout_pts
        if np.any(area_mask):
            s_area = int(math.ceil(np.max(num_pts_in_window) / num_cutout_pts))
            num_ct_pts_area = s_area * num_cutout_pts
            delta_alpha_area = 2.0 * half_alpha / (num_ct_pts_area - 1)
            ang_ct_area = (
                scan_phi[::stride]
                - half_alpha
                + np.arange(num_ct_pts_area, dtype=np.float32).reshape(num_ct_pts_area, 1, 1) * delta_alpha_area
            )
            ang_ct_area = (ang_ct_area + np.pi) % (2.0 * np.pi) - np.pi
            inds_ct_area = (ang_ct_area - scan_phi[0]) / (scan_phi[1] - scan_phi[0])
            inds_ct_area = np.rint(_clip(inds_ct_area, 0, num_pts - 1)).astype(np.int32)

            ct_area = np.take(scans, inds_ct_area + inds_offset)
            ct_area = ct_area.reshape(num_cutout_pts, s_area, num_scans, dists.shape[1]).mean(axis=1)
            ct[:, area_mask] = ct_area[:, area_mask]

    ct[outbound_mask] = padding_val
    ct = _clip(ct, dists - window_depth, dists + window_depth)

    if centered:
        ct = (ct - dists) / window_depth

    return np.ascontiguousarray(ct.transpose((2, 1, 0)), dtype=np.float32)