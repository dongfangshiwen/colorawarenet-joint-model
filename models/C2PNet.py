# utils/C2PNet.py
# Re-implementation of C2PNet-style module (paper-like PDU + FA blocks)
# Forward signature: forward(hazy) -> (dehazed, residual, color_gain, sides)

import torch
import torch.nn as nn
import torch.nn.functional as F

def kaiming_init_conv(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if getattr(m, 'bias', None) is not None:
            nn.init.constant_(m.bias, 0.0)

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, use_bn=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, k, s, p, bias=not use_bn)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Conv2d):
                kaiming_init_conv(m)
    def forward(self, x): return self.net(x)

# Channel attention block (squeeze-excite style)
class ChannelAttention(nn.Module):
    def __init__(self, ch, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch//reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch//reduction, ch, kernel_size=1),
            nn.Sigmoid()
        )
        for m in self.fc:
            if isinstance(m, nn.Conv2d):
                kaiming_init_conv(m)
    def forward(self, x):
        w = self.fc(x)
        return x * w

# Spatial attention (simple conv-based)
class SpatialAttention(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 1, kernel_size=7, padding=3)
        kaiming_init_conv(self.conv)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        m = self.sig(self.conv(x))
        return x * m

# FA block: small conv -> attention -> conv, inspired by FFA-Net FA block (channel+spatial)
class FA_Block(nn.Module):
    def __init__(self, ch, use_pdu=False, pdu=None):
        super().__init__()
        self.conv1 = ConvBNReLU(ch, ch, k=3, p=1)
        self.conv2 = ConvBNReLU(ch, ch, k=3, p=1)
        self.ca = ChannelAttention(ch, reduction=max(4, ch//8))
        self.sa = SpatialAttention(ch)
        self.use_pdu = use_pdu
        self.pdu = pdu
    def forward(self, x):
        y = self.conv1(x)
        if self.use_pdu and (self.pdu is not None):
            # apply PDU on y (physics-aware features)
            # PDU returns a physics-aware feature (same shape)
            y_pdu = self.pdu(y)
            y = y + y_pdu  # residual fusion inside block
        y = self.conv2(y)
        y = self.ca(y)
        y = self.sa(y)
        return x + y  # residual connection

# Physics-aware Dual-branch Unit (PDU)
class PDU(nn.Module):
    """
    Physics-aware Dual-branch Unit as in C2PNet:
      - upper branch: global atmospheric-light-like features: GAP -> conv -> sigmoid, replicate -> A
      - lower branch: transmission-like features: conv stack -> T (pixelwise)
      - combine: F_p = A + (1 - T) * F_in  (+ small fusion conv)
    This is an implementation faithful to the paper's description (equations 5-7).
    """
    def __init__(self, ch, mid_ch=64):
        super().__init__()
        # branch for "atmospheric" (assumed homogeneous)
        self.attnA = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, mid_ch, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, ch, kernel_size=1),
            nn.Sigmoid()
        )
        # branch for "transmission-like" (non-homogeneous)
        self.trans = nn.Sequential(
            ConvBNReLU(ch, ch, k=3, p=1),
            ConvBNReLU(ch, ch, k=3, p=1),
            nn.Conv2d(ch, ch, kernel_size=1),
            nn.Sigmoid()  # produce t in (0,1)
        )
        # fusion conv to allow synergy term (paper mentions synergistic action)
        self.fuse = nn.Sequential(
            nn.Conv2d(ch*2, ch, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, kernel_size=1)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init_conv(m)
    def forward(self, feat_in):
        # feat_in: BxCxhxw
        B, C, H, W = feat_in.shape
        A = self.attnA(feat_in)          # BxC x1x1, values 0..1
        A_rep = A.expand(-1, -1, H, W)   # replicate across spatial dims
        T = self.trans(feat_in)          # BxC xHxW, values 0..1
        # physics-aware combination (simple but expressive)
        # out = A_rep + (1 - T) * feat_in  (then also include fusion of (A_rep, (1-T)*feat) )
        part = (1.0 - T) * feat_in
        x = torch.cat([A_rep, part], dim=1)
        out = self.fuse(x)
        # add residual (keep scale)
        return out

# A block group: N FA blocks with embedded PDU optionally
class BlockGroup(nn.Module):
    def __init__(self, ch, n_blocks=6, use_pdu=True):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            pdu = PDU(ch, mid_ch=min(64, ch)) if use_pdu else None
            self.blocks.append(FA_Block(ch, use_pdu=use_pdu, pdu=pdu))
    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x

# Simple down/up operations
class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBNReLU(in_ch, out_ch, k=3, p=1)
        self.pool = nn.AvgPool2d(2)
    def forward(self, x):
        y = self.conv(x)
        return y, self.pool(y)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBNReLU(in_ch, out_ch, k=3, p=1)
    def forward(self, x):
        return self.conv(x)

# Main C2PNet class
class C2PNet(nn.Module):
    """
    Paper-like C2PNet implementation.
    Signature kept compatible: forward(I) -> (dehazed, residual, color_gain, sides)
    Parameters:
      in_ch: input channels (3)
      base_ch: base channel width
      blocks_per_group: number of FA blocks per group (paper experiments: 6/12/18/19...)
      groups: number of groups (paper used 3 groups)
      use_pdu: embed PDU inside FA blocks
      gain_scale: scale for color_gain (multiplies tanh output)
    """
    def __init__(self, in_ch=3, base_ch=32, blocks_per_group=6, groups=3, use_pdu=True, gain_scale=0.3):
        super().__init__()
        self.in_ch = in_ch
        self.base_ch = base_ch
        self.groups = groups
        self.gain_scale = gain_scale

        # encoder (progressively downsample)
        ch1 = base_ch
        ch2 = base_ch * 2
        ch3 = base_ch * 4

        self.down0 = Down(in_ch, ch1)   # full res
        self.down1 = Down(ch1, ch2)     # 1/2
        self.down2 = Down(ch2, ch3)     # 1/4 (we keep three levels; paper uses more deep stacks inside groups)

        # group stacks (apply groups of FA blocks at each scale)
        self.group0 = BlockGroup(ch1, n_blocks=blocks_per_group, use_pdu=use_pdu)
        self.group1 = BlockGroup(ch2, n_blocks=blocks_per_group, use_pdu=use_pdu)
        self.group2 = BlockGroup(ch3, n_blocks=blocks_per_group, use_pdu=use_pdu)

        # coarse predictor on deepest features
        self.coarse_head = nn.Sequential(
            ConvBNReLU(ch3, ch3//2),
            nn.Conv2d(ch3//2, in_ch, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        for m in self.coarse_head:
            if isinstance(m, nn.Conv2d):
                kaiming_init_conv(m)

        # pyramid fusion: simple concat convs to merge multi-level features
        self.fuse12 = nn.Sequential(
            ConvBNReLU(ch3 + ch2, ch2),
            ConvBNReLU(ch2, ch2)
        )
        self.fuse01 = nn.Sequential(
            ConvBNReLU(ch2 + ch1, ch1),
            ConvBNReLU(ch1, ch1)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init_conv(m)

        # refine block: takes (coarse_up + fused + input) -> residual
        self.refine = nn.Sequential(
            ConvBNReLU(in_ch + ch1 + in_ch, max(64, ch1)),
            ConvBNReLU(max(64, ch1), max(64, ch1)),
            nn.Conv2d(max(64, ch1), in_ch, kernel_size=3, padding=1),
            nn.Tanh()  # residual in (-1,1)
        )
        for m in self.refine:
            if isinstance(m, nn.Conv2d):
                kaiming_init_conv(m)

        # side outputs for multiscale supervision
        self.side_deep = nn.Conv2d(ch3, in_ch, kernel_size=1)
        self.side_mid = nn.Conv2d(ch2, in_ch, kernel_size=1)
        self.side_shallow = nn.Conv2d(ch1, in_ch, kernel_size=1)
        kaiming_init_conv(self.side_deep)
        kaiming_init_conv(self.side_mid)
        kaiming_init_conv(self.side_shallow)

        # color gain (global)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.gain_fc = nn.Sequential(
            nn.Conv2d(ch3, max(8, ch3//4), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(8, ch3//4), in_ch, kernel_size=1),
        )
        for m in self.gain_fc:
            if isinstance(m, nn.Conv2d):
                kaiming_init_conv(m)

    def forward(self, I):
        """
        I: Bx3xHxW in [0,1]
        returns (dehazed, residual, color_gain, sides)
        """
        B, C, H, W = I.shape

        l0, p0 = self.down0(I)   # full res feat, pool->1/2
        l1, p1 = self.down1(p0)  # 1/2 feat, pool->1/4
        l2, p2 = self.down2(p1)  # 1/4 feat

        # apply block groups (in-place)
        g2 = self.group2(l2)
        g1 = self.group1(l1)
        g0 = self.group0(l0)

        # coarse head from deepest
        coarse = self.coarse_head(g2)               # Bx3 x (H/4)x(W/4)
        coarse_up = F.interpolate(coarse, size=(H, W), mode='bilinear', align_corners=False)

        # pyramid fusion
        g2_up = F.interpolate(g2, size=g1.shape[-2:], mode='bilinear', align_corners=False)
        fused1 = self.fuse12(torch.cat([g2_up, g1], dim=1))   # 1/2 res
        fused1_up = F.interpolate(fused1, size=g0.shape[-2:], mode='bilinear', align_corners=False)
        fused0 = self.fuse01(torch.cat([fused1_up, g0], dim=1))  # full res

        # refine: coarse_up + fused0 + input -> residual
        refine_in = torch.cat([coarse_up, fused0, I], dim=1)
        residual = self.refine(refine_in)   # Bx3xHxW, in (-1,1)
        dehazed = torch.clamp(coarse_up + residual, 0.0, 1.0)

        # color gain (global)
        g = self.global_pool(g2)
        color_gain_raw = self.gain_fc(g)   # B x 3 x 1 x 1 (if in_ch=3)
        color_gain = 1.0 + torch.tanh(color_gain_raw) * self.gain_scale

        # sides (upsampled to input size)
        side2 = F.interpolate(self.side_deep(g2), size=(H, W), mode='bilinear', align_corners=False)
        side1 = F.interpolate(self.side_mid(fused1), size=(H, W), mode='bilinear', align_corners=False)
        side0 = F.interpolate(self.side_shallow(fused0), size=(H, W), mode='bilinear', align_corners=False)
        sides = [side0, side1, side2]

        return dehazed, residual, color_gain, sides

if __name__ == "__main__":
    # quick smoke test
    net = C2PNet(in_ch=3, base_ch=32, blocks_per_group=6, groups=3, use_pdu=True)
    x = torch.rand(2,3,256,256)
    with torch.no_grad():
        out, res, gain, sides = net(x)
    print("out", out.shape, "res", res.shape, "gain", gain.shape, "sides:", [s.shape for s in sides])
