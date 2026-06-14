import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.build_sam import build_sam2


# DBlock_DAT相关模块
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class MultiScaleConv(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch3x3 = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels)
        self.branch5x5 = nn.Conv2d(in_channels, in_channels, 5, padding=2, groups=in_channels)
        self.branch7x7 = nn.Conv2d(in_channels, in_channels, 7, padding=3, groups=in_channels)
        self.fuse = nn.Conv2d(in_channels * 3, in_channels, 1)

    def forward(self, x):
        return self.fuse(torch.cat([
            self.branch3x3(x),
            self.branch5x5(x),
            self.branch7x7(x)
        ], dim=1))


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        ctx.eps = eps
        return weight.view(1, -1, 1, 1) * y + bias.view(1, -1, 1, 1)

    @staticmethod
    def backward(ctx, grad_output):
        y, var, weight = ctx.saved_tensors
        eps = ctx.eps
        g = grad_output * weight.view(1, -1, 1, 1)
        mean_g = g.mean(1, keepdim=True)
        mean_gy = (g * y).mean(1, keepdim=True)
        gx = (1. / torch.sqrt(var + eps)) * (g - y * mean_gy - mean_g)
        return gx, grad_output.sum([0, 2, 3]), grad_output.sum([0, 2, 3]), None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class DynamicAdaptiveTanh(nn.Module):
    def __init__(self, normalized_shape, channels_last=True):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.channels_last = channels_last

        # ä¸ºæ¯ä¸ªé€šé"å•ç‹¬å­¦ä¹  alpha å‚æ•°
        self.alpha = nn.Parameter(torch.ones(normalized_shape))
        # ä¸ºæ¯ä¸ªé€šé"å•ç‹¬å­¦ä¹ åç½®å‚æ•°
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        # ç"¨äºŽåŠ¨æ€è°ƒæ•´ alpha çš„ç¼©æ"¾å› å­
        self.alpha_scale = nn.Parameter(torch.ones(1))
        # ç"¨äºŽåŠ¨æ€è°ƒæ•´åç½®çš„ç¼©æ"¾å› å­
        self.bias_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # è®¡ç®—è¾"å…¥æ•°æ®çš„å‡å€¼å'Œæ ‡å‡†å·®
        if self.channels_last:
            mean = torch.mean(x, dim=(0, 2, 3), keepdim=True)
            std = torch.std(x, dim=(0, 2, 3), keepdim=True)
        else:
            mean = torch.mean(x, dim=(0, 2, 3), keepdim=True)
            std = torch.std(x, dim=(0, 2, 3), keepdim=True)

        # æ ¹æ®è¾"å…¥æ•°æ®çš„ç»Ÿè®¡ç‰¹å¾åŠ¨æ€è°ƒæ•´ alpha å'Œåç½®
        dynamic_alpha = self.alpha_scale * self.alpha.view(1, -1, 1, 1) / (std + 1e-8)
        dynamic_bias = self.bias_scale * self.bias.view(1, -1, 1, 1) * mean

        x = torch.tanh(dynamic_alpha * x + dynamic_bias)

        if self.channels_last:
            return x
        else:
            return x


class DBlock_DAT(nn.Module):
    def __init__(self, in_c, out_c, DW_Expand=2, FFN_Expand=2, extra_depth_wise=True):
        super().__init__()
        # ä½¿ç"¨è¾"å‡ºé€šé"æ•°ä½œä¸ºå†…éƒ¨å¤„ç†é€šé"æ•°
        c = out_c
        self.dw_channel = DW_Expand * c
        
        # è¾"å…¥é€šé"é€‚é…
        self.input_proj = nn.Conv2d(in_c, c, 1) if in_c != c else nn.Identity()
        
        self.conv1 = nn.Conv2d(c, self.dw_channel, 1)
        self.extra_conv = nn.Conv2d(self.dw_channel, self.dw_channel, 3, padding=1, groups=c) if extra_depth_wise else nn.Identity()

        self.msconv = MultiScaleConv(self.dw_channel)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dw_channel // 2, self.dw_channel // 2, 1)
        )
        self.sg1 = SimpleGate()
        
        # åœ¨sg1ä¹‹åŽæ·»åŠ DAT
        self.dat = DynamicAdaptiveTanh([self.dw_channel // 2])
        
        self.sg2 = SimpleGate()
        self.conv3 = nn.Conv2d(self.dw_channel // 2, c, 1)

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)))
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)))

    def forward(self, inp, adapter=None):
        # è¾"å…¥é€šé"é€‚é…
        inp = self.input_proj(inp)
        
        y = inp
        x = self.norm1(inp)
        x = self.extra_conv(self.conv1(x))
        z = self.msconv(x)
        z = self.sg1(z)
        
        # åœ¨sg1ä¹‹åŽåº"ç"¨DAT
        z = self.dat(z)
        
        x = self.sca(z) * z
        x = self.conv3(x)
        y = inp + self.beta * x

        x = self.conv4(self.norm2(y))
        x = self.sg2(x)
        x = self.conv5(x)
        return y + x * self.gamma


# FusedEnhanceBlock相关模块
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super(ChannelAttention, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.pool(x)
        w = self.fc(w)
        return x * w


class FreMLP(nn.Module):
    def __init__(self, nc, expand=2):
        super(FreMLP, self).__init__()
        self.process_mag = nn.Sequential(
            nn.Conv2d(nc, expand * nc, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(expand * nc, nc, 1, 1, 0)
        )

    def forward(self, x):
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x, norm='backward')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)
        mag = self.process_mag(mag)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        return x_out


class FusedEnhanceBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, dilations=[1, 4, 9]):
        super(FusedEnhanceBlock, self).__init__()
        self.dw_channel = DW_Expand * c

        self.conv1 = nn.Conv2d(c, self.dw_channel, kernel_size=1)
        self.branches = nn.ModuleList([
            nn.Conv2d(self.dw_channel, self.dw_channel, 3, padding=d, dilation=d, groups=self.dw_channel)
            for d in dilations
        ])

        self.sg = SimpleGate()
        self.ca = ChannelAttention(self.dw_channel // 2)
        self.conv2 = nn.Conv2d(self.dw_channel // 2, c, kernel_size=1)

        self.norm1 = nn.BatchNorm2d(c)
        self.norm2 = nn.BatchNorm2d(c)

        self.freq = FreMLP(c)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)))
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)))

    def forward(self, x):
        y = x
        x = self.norm1(x)
        x = self.conv1(x)
        z = sum([branch(x) for branch in self.branches])
        z = self.sg(z)
        z = self.ca(z)
        z = self.conv2(z)
        y = y + self.beta * z

        x_freq = self.freq(self.norm2(y))
        out = y + self.gamma * (y * x_freq)
        return out


# 原有模块
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
    
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Adapter(nn.Module):
    def __init__(self, blk) -> None:
        super(Adapter, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU()
        )

    def forward(self, x):
        prompt = self.prompt_learn(x)
        promped = x + prompt
        net = self.block(promped)
        return net
    

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
    

class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class SAM2UNet(nn.Module):
    def __init__(self, checkpoint_path=None) -> None:
        super(SAM2UNet, self).__init__()    
        model_cfg = "sam2_hiera_l.yaml"
        if checkpoint_path:
            model = build_sam2(model_cfg, checkpoint_path)
        else:
            model = build_sam2(model_cfg)
        del model.sam_mask_decoder
        del model.sam_prompt_encoder
        del model.memory_encoder
        del model.memory_attention
        del model.mask_downsample
        del model.obj_ptr_tpos_proj
        del model.obj_ptr_proj
        del model.image_encoder.neck
        self.encoder = model.image_encoder.trunk

        for param in self.encoder.parameters():
            param.requires_grad = False
        blocks = []
        for block in self.encoder.blocks:
            blocks.append(
                Adapter(block)
            )
        self.encoder.blocks = nn.Sequential(
            *blocks
        )
        self.rfb1 = DBlock_DAT(144, 64)
        self.rfb2 = DBlock_DAT(288, 64)
        self.rfb3 = DBlock_DAT(576, 64)
        self.rfb4 = DBlock_DAT(1152, 64)
        self.up1 = (Up(128, 64))
        self.up2 = (Up(128, 64))
        self.up3 = (Up(128, 64))
        self.up4 = (Up(128, 64))
        
        # 在RFB处理后添加FusedEnhanceBlock
        self.fused1 = FusedEnhanceBlock(64)
        self.fused2 = FusedEnhanceBlock(64)
        self.fused3 = FusedEnhanceBlock(64)
        self.fused4 = FusedEnhanceBlock(64)
        
        self.side1 = nn.Conv2d(64, 1, kernel_size=1)
        self.side2 = nn.Conv2d(64, 1, kernel_size=1)
        self.head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        x1, x2, x3, x4 = self.rfb1(x1), self.rfb2(x2), self.rfb3(x3), self.rfb4(x4)
        
        # 在RFB处理后应用FusedEnhanceBlock
        x1, x2, x3, x4 = self.fused1(x1), self.fused2(x2), self.fused3(x3), self.fused4(x4)
        
        x = self.up1(x4, x3)
        out1 = F.interpolate(self.side1(x), scale_factor=16, mode='bilinear')
        x = self.up2(x, x2)
        out2 = F.interpolate(self.side2(x), scale_factor=8, mode='bilinear')
        x = self.up3(x, x1)
        out = F.interpolate(self.head(x), scale_factor=4, mode='bilinear')
        return out, out1, out2


if __name__ == "__main__":
    with torch.no_grad():
        model = SAM2UNet().cuda()
        x = torch.randn(1, 3, 352, 352).cuda()
        out, out1, out2 = model(x)
        print(out.shape, out1.shape, out2.shape)