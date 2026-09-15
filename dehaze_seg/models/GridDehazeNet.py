# Grid dehazing model.
# -*- coding: utf-8 -*-
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Utilities / basic blocks
# -------------------------
def kaiming_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if getattr(m, 'bias', None) is not None:
            nn.init.constant_(m.bias, 0.0)


class BasicResBlock(nn.Module):
    """Simple residual block: conv3x3 -> BN (optional) -> ReLU -> conv3x3 -> BN -> add -> ReLU"""
    def __init__(self, ch, norm=True):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=True))
        if norm:
            layers.append(nn.BatchNorm2d(ch))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=True))
        if norm:
            layers.append(nn.BatchNorm2d(ch))
        self.body = nn.Sequential(*layers)
        kaiming_init(self.body[0])
        # init second conv
        for m in self.body:
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)

    def forward(self, x):
        return F.relu(x + self.body(x))


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block"""
    def __init__(self, ch, r=8):
        super().__init__()
        self.fc1 = nn.Conv2d(ch, max(1, ch // r), kernel_size=1)
        self.fc2 = nn.Conv2d(max(1, ch // r), ch, kernel_size=1)
        kaiming_init(self.fc1); kaiming_init(self.fc2)
    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s


class SpatialAttention(nn.Module):
    """Simple spatial attention: conv -> sigmoid"""
    def __init__(self, in_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 1, kernel_size=7, padding=3, bias=True)
        kaiming_init(self.conv)
    def forward(self, x):
        att = torch.sigmoid(self.conv(x))
        return x * att


class SCSE(nn.Module):
    """Concurrent spatial and channel squeeze & excitation"""
    def __init__(self, ch, r=8):
        super().__init__()
        self.cSE = SEBlock(ch, r=r)
        self.sSE = SpatialAttention(ch)
    def forward(self, x):
        # sum of cSE and sSE branch (as in some implementations)
        return self.cSE(x) + self.sSE(x)


class DownSample(nn.Module):
    """Downsample by factor 2: conv stride 2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
        kaiming_init(self.conv)
    def forward(self, x):
        return F.relu(self.conv(x))


class UpSample(nn.Module):
    """Upsample by factor 2 then conv"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        kaiming_init(self.conv)
    def forward(self, x, size):
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        return F.relu(self.conv(x))


# -------------------------
# GridDehazeNet
# -------------------------
class GridDehazeNet(nn.Module):
    """
    GridDehazeNet-like module with padding/truncation fix.

    Parameters:
        in_ch: input image channels (3)
        rows: number of scales (>=2)
        cols: grid columns (depth)
        base_ch: base channel count at top scale
        use_scse: whether to apply SCSE in node proc
    """
    def __init__(self, in_ch: int = 3, rows: int = 3, cols: int = 6, base_ch: int = 32, use_scse: bool = True):
        super().__init__()
        assert rows >= 2, "rows must be >= 2"
        self.in_ch = in_ch
        self.rows = rows
        self.cols = cols
        self.base_ch = base_ch
        self.use_scse = use_scse

        # Preprocessing: shallow conv stack
        self.pre_conv = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        kaiming_init(self.pre_conv[0]); kaiming_init(self.pre_conv[2])

        # scale channels (top->low: base_ch, base_ch*2, ...)
        self.scale_chs = [base_ch * (2 ** i) for i in range(rows)]

        # initial converters to get features at each scale for column 0
        self.init_scale_convs = nn.ModuleList()
        for r in range(rows):
            if r == 0:
                self.init_scale_convs.append(nn.Identity())
            else:
                conv = nn.Sequential(
                    nn.Conv2d(self.scale_chs[r-1], self.scale_chs[r], kernel_size=3, stride=2, padding=1, bias=True),
                    nn.ReLU(inplace=True)
                )
                kaiming_init(conv[0])
                self.init_scale_convs.append(conv)

        # grid node modules and bookkeeping of expected in channels
        self.grid_blocks = nn.ModuleDict()
        self.node_in_ch = {}  # expected maximum input channels for each node (used for pad/truncate)
        for c in range(cols):
            for r in range(rows):
                ch = self.scale_chs[r]
                node_name = f"g_{r}_{c}"
                # For column 0 we only expect single input (the initialized feature)
                if c == 0:
                    expected_in = ch
                else:
                    # internal nodes may receive left + up/down => up to 3*ch
                    expected_in = ch * 3
                # reduction conv: accept expected_in channels -> project to ch
                reduce_conv = nn.Conv2d(expected_in, ch, kernel_size=1, bias=True)
                kaiming_init(reduce_conv)
                proc = nn.Sequential(
                    reduce_conv,
                    nn.ReLU(inplace=True),
                    BasicResBlock(ch),
                )
                if use_scse:
                    # wrap so proc remains a Sequential
                    proc = nn.Sequential(proc, SCSE(ch))
                self.grid_blocks[node_name] = proc
                self.node_in_ch[node_name] = expected_in

        # down/upsampling operators for vertical connections
        self.down_ops = nn.ModuleDict()
        self.up_ops = nn.ModuleDict()
        for r in range(rows - 1):
            self.down_ops[f"d_{r}"] = DownSample(self.scale_chs[r], self.scale_chs[r+1])
            self.up_ops[f"u_{r}"] = UpSample(self.scale_chs[r+1], self.scale_chs[r])

        # post-processing: from top-scale final feature -> residual RGB
        top_ch = self.scale_chs[0]
        self.post = nn.Sequential(
            nn.Conv2d(top_ch, top_ch, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(top_ch, in_ch, kernel_size=3, padding=1, bias=True)
        )
        kaiming_init(self.post[0]); kaiming_init(self.post[2])
        # zero-init last conv bias to encourage small residual initially
        try:
            nn.init.constant_(self.post[-1].bias, 0.0)
        except Exception:
            pass

        # side outputs heads
        self.side_heads = nn.ModuleList()
        for r in range(rows):
            head = nn.Conv2d(self.scale_chs[r], in_ch, kernel_size=3, padding=1, bias=True)
            kaiming_init(head)
            self.side_heads.append(head)

    def _init_grid_state(self, x: torch.Tensor) -> List[List[Optional[torch.Tensor]]]:
        """Initialize states for column 0 at each scale."""
        states = [[None for _ in range(self.cols)] for __ in range(self.rows)]
        top = self.pre_conv(x)  # B x base_ch x H x W
        states[0][0] = top
        for r in range(1, self.rows):
            prev = states[r-1][0]
            states[r][0] = self.init_scale_convs[r](prev)
        return states

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Optional[torch.Tensor]]]:
        """
        x: B x 3 x H x W (assumed in [0,1])
        returns: dehazed, residual, color_gain, sides
        """
        B, C, H, W = x.shape
        states = self._init_grid_state(x)

        # iterate columns (we already have column 0 initialized)
        for c in range(1, self.cols):
            for r in range(self.rows):
                node_name = f"g_{r}_{c}"
                left = states[r][c-1]
                inputs = []
                if left is not None:
                    inputs.append(left)
                # upper neighbor (from r-1) downsampled to this scale
                if r - 1 >= 0:
                    up = states[r-1][c-1]
                    if up is not None:
                        up_ds = self.down_ops[f"d_{r-1}"](up)
                        inputs.append(up_ds)
                # lower neighbor (from r+1) upsampled to this scale
                if r + 1 < self.rows:
                    down = states[r+1][c-1]
                    if down is not None:
                        target_h = left.shape[-2] if left is not None else down.shape[-2]*2
                        target_w = left.shape[-1] if left is not None else down.shape[-1]*2
                        down_us = self.up_ops[f"u_{r}"](down, size=(target_h, target_w))
                        inputs.append(down_us)

                if len(inputs) == 0:
                    # fallback: copy previous column same-row if exists
                    states[r][c] = states[r][c-1]
                    continue

                # align spatial sizes (use first as ref)
                ref = inputs[0]
                aligned = []
                for t in inputs:
                    if t.shape[-2:] != ref.shape[-2:]:
                        t = F.interpolate(t, size=ref.shape[-2:], mode='bilinear', align_corners=False)
                    aligned.append(t)
                concat = torch.cat(aligned, dim=1) if len(aligned) > 1 else aligned[0]

                # ensure concat channels match node expected input channels (pad zeros or truncate)
                expected_ch = self.node_in_ch[node_name]
                cur_ch = concat.shape[1]
                if cur_ch < expected_ch:
                    pad_ch = expected_ch - cur_ch
                    zeros = torch.zeros(concat.size(0), pad_ch, concat.size(2), concat.size(3), dtype=concat.dtype, device=concat.device)
                    concat = torch.cat([concat, zeros], dim=1)
                elif cur_ch > expected_ch:
                    concat = concat[:, :expected_ch, :, :]

                proc = self.grid_blocks[node_name]
                out = proc(concat)
                states[r][c] = out

        # take final top-scale feature
        top_feat = states[0][self.cols - 1]
        residual = self.post(top_feat)
        dehazed = torch.clamp(x + residual, 0.0, 1.0)

        # side outputs (upsample to input resolution)
        sides = []
        for r, head in enumerate(self.side_heads):
            feat = states[r][self.cols - 1]
            s = head(feat)
            if s.shape[-2:] != (H, W):
                s = F.interpolate(s, size=(H, W), mode='bilinear', align_corners=False)
            sides.append(s)

        color_gain = torch.ones(dehazed.shape[0], dehazed.shape[1], 1, 1, device=dehazed.device, dtype=dehazed.dtype)
        return dehazed, residual, color_gain, sides


# -------------------------
# Smoke test
# -------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m = GridDehazeNet(in_ch=3, rows=3, cols=6, base_ch=32, use_scse=True).to(device)
    x = torch.rand(2, 3, 256, 256, device=device)
    with torch.no_grad():
        out, res, gain, sides = m(x)
    print("out", out.shape, "res", res.shape, "gain", gain.shape)
    print("sides shapes:", [s.shape if s is not None else None for s in sides])
    print("params:", sum(p.numel() for p in m.parameters() if p.requires_grad))
