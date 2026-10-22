"""LHRF-YOLO modules.

Implements the components described in "LHRF-YOLO: A Lightweight Model with
Hybrid Receptive Field for Forest Fire Detection" (Forests 2025, 16, 1095):

- SSBlock / SS2D : selective scan block with global context modelling.
- RMELAN         : residual multi-branch efficient layer aggregation network.
- DEPMD          : dynamic enhanced patch-merge downsampling.
- SWF            : scaling weighted fusion.
"""

import math

import torch
import torch.nn as nn
from einops import rearrange

from .block import Bottleneck
from .conv import Conv
from .csm_triton import cross_merge_fn, cross_scan_fn
from .csms6s import selective_scan_fn

__all__ = ("RMELAN", "DEPMD", "SWF", "SSBlock", "SS2D")


class SEBlock(nn.Module):
    """Squeeze-and-excitation block used by DEPMD."""

    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.Mish(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.avg_pool(x))


class CGLU(nn.Module):
    """Convolutional gated linear unit used in SSBlock."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = int(2 * (hidden_features or in_features) / 3)
        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, 1)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x, v = self.fc1(x).chunk(2, dim=1)
        x = self.drop(self.act(self.dwconv(x) + x) * v)
        return self.drop(self.fc2(x))


class MambaInit:
    """Parameter initialisation for the selective scan."""

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        else:
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)

        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).view(1, -1).repeat(d_inner, 1)
        A_log = torch.log(A)
        if copies > 0:
            A_log = A_log[None].repeat(copies, 1, 1)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log.contiguous())
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = D[None].repeat(copies, 1)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    @classmethod
    def init_dt_A_D(cls, d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=4):
        dt_projs = [cls.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor) for _ in range(k_group)]
        dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in dt_projs], dim=0))
        dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in dt_projs], dim=0))
        del dt_projs

        A_logs = cls.A_log_init(d_state, d_inner, copies=k_group, merge=True)
        Ds = cls.D_init(d_inner, copies=k_group, merge=True)
        return A_logs, Ds, dt_projs_weight, dt_projs_bias


class SS2D(nn.Module):
    """2D selective scan (cross-scan, selective scan, cross-merge)."""

    def __init__(
        self,
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.Mish,
        d_conv=3,
        conv_bias=True,
        dropout=0.0,
        bias=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        **kwargs,
    ):
        super().__init__()
        self.k_group = 4
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = int(math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank)
        self.with_dconv = d_conv > 1

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        self.act = act_layer()
        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                self.d_inner,
                self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
            )

        x_proj = [nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False) for _ in range(self.k_group)]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in x_proj], dim=0))
        del x_proj

        self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = MambaInit.init_dt_A_D(
            self.d_state, self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
            k_group=self.k_group,
        )

        self.out_act = nn.Identity()
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward_core(self, x):
        B, D, H, W = x.shape
        N = self.d_state
        K, R = self.k_group, self.dt_rank
        L = H * W

        xs = cross_scan_fn(x, in_channel_first=True, out_channel_first=True, scans=0)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.view(B, -1, L)
        dts = dts.contiguous().view(B, -1, L)
        As = -self.A_logs.float().exp()
        Ds = self.Ds.float()
        Bs = Bs.contiguous().view(B, K, N, L)
        Cs = Cs.contiguous().view(B, K, N, L)
        delta_bias = self.dt_projs_bias.view(-1).float()

        xs, dts, Bs, Cs = xs.float(), dts.float(), Bs.float(), Cs.float()
        ys = selective_scan_fn(xs, dts, As, Bs, Cs, Ds, delta_bias, True, True, backend="oflex")
        ys = ys.view(B, K, -1, H, W)

        y = cross_merge_fn(ys, in_channel_first=True, out_channel_first=True, scans=0)
        y = y.view(B, -1, H * W).transpose(1, 2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        return y.to(x.dtype)

    def forward(self, x):
        x = self.in_proj(x)
        x, z = x.chunk(2, dim=-1)
        z = self.act(z)
        x = x.permute(0, 3, 1, 2).contiguous()
        if self.with_dconv:
            x = self.conv2d(x)
        x = self.act(x)

        y = self.out_act(self.forward_core(x)) * z
        return self.dropout(self.out_proj(y))

    # Names kept for checkpoints saved by the research code.
    def forwardv2(self, x, **kwargs):
        return self.forward(x)

    def forward_corev2(self, x, **kwargs):
        return self.forward_core(x)


class SSBlock(nn.Module):
    """Selective scan block: LayerNorm -> SS2D -> LayerNorm -> CGLU with residuals."""

    def __init__(self, in_channels, hidden_dim=0, ssm_d_state=16, ffn_ratio=2, ssm_ratio=2.0, ssm_dt_rank="auto",
                 ssm_act_layer=nn.Mish, ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0, **kwargs):
        super().__init__()
        self.op = SS2D(
            d_model=hidden_dim,
            d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=ssm_dt_rank,
            act_layer=ssm_act_layer,
            d_conv=ssm_conv,
            conv_bias=ssm_conv_bias,
            dropout=ssm_drop_rate,
        )
        self.ffn = CGLU(in_features=in_channels, hidden_features=int(in_channels * ffn_ratio))
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x1 = rearrange(x, "b c h w -> b h w c").contiguous()
        x1 = rearrange(self.op(self.norm1(x1)), "b h w c -> b c h w").contiguous() + x
        x2 = rearrange(x1, "b c h w -> b h w c").contiguous()
        x2 = rearrange(self.norm2(x2), "b h w c -> b c h w").contiguous()
        return x1 + self.ffn(x2)


class ChannelGate(nn.Module):
    """Channel-wise gate used inside RMELAN to re-weight aggregated branches."""

    def __init__(self, channels, reduction=16, init_tau=5.0, tau_decay=0.97, batches_per_epoch=189):
        super().__init__()
        self.channels = channels
        self.tau_decay = tau_decay
        self.batches_per_epoch = batches_per_epoch

        self.gate_conv = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.Mish(),
            nn.Conv2d(channels // reduction, channels, 1),
        )
        self.register_buffer("tau", torch.tensor(float(init_tau)))
        self.register_buffer("min_tau", torch.tensor(0.5))
        self.register_buffer("batch_counter", torch.tensor(0))
        self.register_buffer("epoch_counter", torch.tensor(0))

        nn.init.normal_(self.gate_conv[-1].weight, std=0.01)
        nn.init.constant_(self.gate_conv[-1].bias, 0.0)

    def _anneal(self):
        if self.training:
            self.batch_counter += 1
            if self.batch_counter >= self.batches_per_epoch:
                with torch.no_grad():
                    self.tau.data *= self.tau_decay
                    self.tau.data.clamp_(min=self.min_tau)
                    self.epoch_counter += 1
                    self.batch_counter.zero_()

    def forward(self, x):
        self._anneal()
        B, C = x.shape[:2]
        theta = self.gate_conv(x.mean(dim=[2, 3], keepdim=True)).view(B, C)

        if self.training:
            noise = -torch.log(-torch.log(torch.rand_like(theta) + 1e-10))
            gate = torch.sigmoid((theta + noise) / self.tau)
            hard_gate = (gate > 0.5).float()
            gate = gate + (hard_gate - gate.detach())
        else:
            gate = (theta / self.tau >= 0.5).float()
        return x * gate.view(B, C, 1, 1)


class RMELAN(nn.Module):
    """Residual multi-branch ELAN (Sec. 2.2.2).

    One branch runs SSBlock -> Bottleneck, the other Bottleneck -> SSBlock.
    Outputs are concatenated with the input projection and added to the input.
    """

    def __init__(self, in_channels, out_channels, n=1, ffn_ratio=4):
        super().__init__()
        self.inc = in_channels
        self.outc = out_channels
        self.split_channels = int(out_channels * 0.25)
        self.num = 3 + 2 * n

        ssm_kwargs = dict(ssm_d_state=1, ssm_ratio=2, ffn_ratio=ffn_ratio)
        self.ss1 = nn.Sequential(SSBlock(self.split_channels, self.split_channels, **ssm_kwargs))
        self.lconv1 = nn.Sequential(Bottleneck(self.split_channels, self.split_channels))

        self.conv1 = Conv(in_channels, self.split_channels)
        self.lconv = nn.Sequential(*[Bottleneck(self.split_channels, self.split_channels) for _ in range(n)])
        self.ss = nn.Sequential(*[SSBlock(self.split_channels, self.split_channels, **ssm_kwargs) for _ in range(n)])

        self.trans = Conv(self.num * self.split_channels, out_channels)
        self.mix = ChannelGate(self.num * self.split_channels)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.ss(x1)
        x3 = self.lconv(x1)
        x4 = self.ss1(x3)
        x5 = self.lconv1(x2)
        out = self.trans(self.mix(torch.cat((x1, x2, x3, x4, x5), 1)))
        return out + x if self.inc == self.outc else out


class DEPMD(nn.Module):
    """Dynamic enhanced patch-merge downsampling (Sec. 2.2.3).

    Patch-merge (2x) projection followed by SE recalibration and a learnable
    scaling factor.
    """

    def __init__(self, dim, out_dim):
        super().__init__()
        self.hidden = int(dim * 4)
        self.pw_linear = nn.Sequential(
            nn.Conv2d(self.hidden, out_dim, 1, 1, 0),
            nn.BatchNorm2d(out_dim),
            nn.Mish(),
        )
        self.se = SEBlock(out_dim)
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x):
        y = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], dim=1)
        y = self.pw_linear(y)
        return y + self.se(y) * self.alpha


class SWF(nn.Module):
    """Scaling weighted fusion for two feature maps (Sec. 2.2.4)."""

    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(2))
        self.v = nn.Parameter(torch.ones(1))
        self.epsilon = 0.0001

    def forward(self, x):
        weight = self.v[0] * self.w / (torch.sum(self.w, dim=0) + self.epsilon)
        return torch.cat([weight[0] * x[0], weight[1] * x[1]], self.d)
