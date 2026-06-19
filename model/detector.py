import numpy as np
import torch

from .model import HumanDetectionModel
from .preprocess import scans_to_cutout
from .postprocess import nms_predicted_center


class Detector:
    def __init__(self, ckpt_file, gpu=False, stride=1, panoramic_scan=False):
        self._gpu = bool(gpu and torch.cuda.is_available())
        self._stride = int(stride)
        self._laser_fov_deg = None
        self._scan_phi = None
        self._scan_size = None

        self._model = HumanDetectionModel(
            dropout=0.5,
            num_pts=56,
            embedding_length=128,
            alpha=0.5,
            window_size=17,
            panoramic_scan=panoramic_scan,
        )

        if self._gpu:
            ckpt = torch.load(ckpt_file)
        else:
            ckpt = torch.load(ckpt_file, map_location=torch.device("cpu"))

        state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        self._model.load_state_dict(state, strict=True)
        self._model.eval()

        if self._gpu:
            torch.backends.cudnn.benchmark = True
            self._model = self._model.cuda()

    def set_laser_fov(self, fov_deg):
        self._laser_fov_deg = float(fov_deg)
        self._scan_phi = None

    def is_ready(self):
        return self._laser_fov_deg is not None

    def _build_scan_phi(self, n_pts):
        half_fov_rad = 0.5 * np.deg2rad(self._laser_fov_deg)
        self._scan_phi = np.linspace(-half_fov_rad, half_fov_rad, n_pts, dtype=np.float32)
        self._scan_size = int(n_pts)

    def __call__(self, scan):
        if not self.is_ready():
            raise RuntimeError("Call set_laser_fov() first.")

        scan = np.asarray(scan, dtype=np.float32)
        if self._scan_phi is None or self._scan_size != len(scan):
            self._build_scan_phi(len(scan))

        ct = scans_to_cutout(
            scan[None, ...],
            self._scan_phi,
            stride=self._stride,
            centered=True,
            fixed=True,
            window_width=1.0,
            window_depth=0.5,
            num_cutout_pts=56,
            padding_val=29.99,
            area_mode=True,
        )

        ct_t = torch.from_numpy(ct).float()
        if self._gpu:
            ct_t = ct_t.cuda(non_blocking=True)

        with torch.no_grad():
            pred_cls, pred_reg, _ = self._model(ct_t.unsqueeze(dim=0), inference=True)

        pred_cls = torch.sigmoid(pred_cls[0]).cpu().numpy()
        pred_reg = pred_reg[0].cpu().numpy()

        dets_xy, dets_cls, instance_mask = nms_predicted_center(
            scan[:: self._stride],
            self._scan_phi[:: self._stride],
            pred_cls[:, 0],
            pred_reg,
            min_dist=0.5,
        )

        return dets_xy, dets_cls, instance_mask