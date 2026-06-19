from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import _conv1d_3


class HumanDetectionModel(nn.Module):
    def __init__(
        self,
        dropout=0.5,
        num_pts=56,
        alpha=0.5,
        embedding_length=128,
        window_size=17,
        panoramic_scan=False,
    ):
        super().__init__()

        self.dropout = dropout

        self.conv_block_1 = nn.Sequential(
            _conv1d_3(1, 64),
            _conv1d_3(64, 64),
            _conv1d_3(64, 128),
        )
        self.conv_block_2 = nn.Sequential(
            _conv1d_3(128, 128),
            _conv1d_3(128, 128),
            _conv1d_3(128, 256),
        )
        self.conv_block_3 = nn.Sequential(
            _conv1d_3(256, 256),
            _conv1d_3(256, 256),
            _conv1d_3(256, 512),
        )
        self.conv_block_4 = nn.Sequential(
            _conv1d_3(512, 256),
            _conv1d_3(256, 128),
        )

        self.conv_cls = nn.Conv1d(128, 1, kernel_size=1)
        self.conv_reg = nn.Conv1d(128, 2, kernel_size=1)

        self.gate = _SpatialAttentionMemory(
            n_pts=int(ceil(num_pts / 4)),
            n_channel=256,
            embedding_length=embedding_length,
            alpha=alpha,
            window_size=window_size,
            panoramic_scan=panoramic_scan,
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, a=0.1, nonlinearity="leaky_relu")
                if getattr(m, "bias", None) is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, inference=False):
        bsz, n_cutout, n_scan, n_pts = x.shape

        if not inference:
            self.gate.reset()

        for i in range(n_scan):
            x_i = x[:, :, i, :]
            out = x_i.reshape(bsz * n_cutout, 1, n_pts)

            out = self._conv_and_pool(out, self.conv_block_1)
            out = self._conv_and_pool(out, self.conv_block_2)

            out = out.reshape(bsz, n_cutout, out.shape[-2], out.shape[-1])
            out, sim = self.gate(out)

            out = out.reshape(bsz * n_cutout, out.shape[-2], out.shape[-1])
            out = self._conv_and_pool(out, self.conv_block_3)
            out = self.conv_block_4(out)
            out = F.avg_pool1d(out, kernel_size=out.shape[-1])

            pred_cls = self.conv_cls(out).reshape(bsz, n_cutout, -1)
            pred_reg = self.conv_reg(out).reshape(bsz, n_cutout, 2)

        return pred_cls, pred_reg, sim

    def _conv_and_pool(self, x, conv_block):
        out = conv_block(x)
        out = F.max_pool1d(out, kernel_size=2)
        if self.dropout > 0:
            out = F.dropout(out, p=self.dropout, training=self.training)
        return out


class _SpatialAttentionMemory(nn.Module):
    def __init__(
        self,
        n_pts,
        n_channel,
        embedding_length,
        alpha,
        window_size,
        panoramic_scan,
    ):
        super().__init__()
        self._alpha = alpha
        self._window_size = window_size
        self._embedding_length = embedding_length
        self._panoramic_scan = panoramic_scan

        self.conv = nn.Sequential(
            nn.Conv1d(n_channel, embedding_length, kernel_size=n_pts, padding=0),
            nn.BatchNorm1d(embedding_length),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self._memory = None
        self.neighbor_masks = None

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, a=0.1, nonlinearity="leaky_relu")
                if getattr(m, "bias", None) is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def reset(self):
        self._memory = None

    def forward(self, x_new):
        if self._memory is None:
            self._memory = x_new
            return self._memory, None

        n_batch, n_cutout, n_channel, n_pts = x_new.shape

        if self.neighbor_masks is None or self.neighbor_masks.shape[0] != n_cutout:
            self.neighbor_masks = self._generate_neighbor_mask(x_new)

        emb_x = self.conv(x_new.reshape(n_batch * n_cutout, n_channel, n_pts))
        emb_x = emb_x.reshape(n_batch, n_cutout, self._embedding_length)

        emb_mem = self.conv(self._memory.reshape(n_batch * n_cutout, n_channel, n_pts))
        emb_mem = emb_mem.reshape(n_batch, n_cutout, self._embedding_length)

        sim = torch.matmul(emb_x, emb_mem.permute(0, 2, 1))
        sim = sim - 1e10 * (1.0 - self.neighbor_masks)
        maxes = sim.max(dim=-1, keepdim=True)[0]
        exps = torch.exp(sim - maxes) * self.neighbor_masks
        sim = exps / exps.sum(dim=-1, keepdim=True)

        atten_memory = self._memory.reshape(n_batch, n_cutout, n_channel * n_pts)
        atten_memory = torch.matmul(sim, atten_memory)
        atten_memory = atten_memory.reshape(n_batch, n_cutout, n_channel, n_pts)

        self._memory = self._alpha * x_new + (1.0 - self._alpha) * atten_memory
        return self._memory, sim

    def _generate_neighbor_mask(self, x):
        n_cutout = x.shape[1]
        hw = int(self._window_size / 2)

        inds_col = torch.arange(n_cutout, device=x.device).unsqueeze(-1).long()
        window_inds = torch.arange(-hw, hw + 1, device=x.device).long()
        inds_col = inds_col + window_inds.unsqueeze(0)

        if self._panoramic_scan and not self.training:
            inds_col = inds_col % n_cutout
        else:
            inds_col = inds_col.clamp(min=0, max=n_cutout - 1)

        inds_row = torch.arange(n_cutout, device=x.device).unsqueeze(-1).expand_as(inds_col).long()
        inds_full = torch.stack((inds_row, inds_col), dim=2).reshape(-1, 2)

        masks = torch.zeros(n_cutout, n_cutout, device=x.device, dtype=torch.float32)
        masks[inds_full[:, 0], inds_full[:, 1]] = 1.0
        return masks