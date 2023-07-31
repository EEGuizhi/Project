""" Full assembly of the parts to form the complete network """
from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)



class ConvBlocks(nn.Module):
    def __init__(self, in_ch, channels, kernel_sizes=None, strides=None, dilations=None, paddings=None, BatchNorm=nn.BatchNorm2d):
        super(ConvBlocks, self).__init__()
        self.num = len(channels)
        if kernel_sizes is None: kernel_sizes = [3 for c in channels]
        if strides is None: strides = [1 for c in channels]
        if dilations is None: dilations = [1 for c in channels]
        if paddings is None:
            paddings = [
                ((kernel_sizes[i] // 2) if dilations[i] == 1 else (kernel_sizes[i] // 2 * dilations[i])) for i in range(self.num)
            ]
        convs_tmp = []
        for i in range(self.num):
            if channels[i] == 1:
                convs_tmp.append(
                    nn.Conv2d(
                        in_ch if i == 0 else channels[i - 1], channels[i], kernel_size=kernel_sizes[i],
                        stride=strides[i], padding=paddings[i], dilation=dilations[i]
                    )
                )
            else:
                convs_tmp.append(nn.Sequential(
                    nn.Conv2d(
                        in_ch if i == 0 else channels[i - 1], channels[i], kernel_size=kernel_sizes[i],
                        stride=strides[i], padding=paddings[i], dilation=dilations[i], bias=False
                    ),
                    BatchNorm(channels[i]),
                    nn.ReLU()
                ))
        self.convs = nn.Sequential(*convs_tmp)

        # weight initialization
        for m in self.convs.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        return self.convs(x)

class IGGNet(nn.Module):  # Interaction-Guided Gating Network (Fc另外寫)
    def __init__(self, in_ch, out_ch, num_classes, SE_maxpool=True, SE_softmax=False):
        super(IGGNet, self).__init__()
        self.hintEncoder = ConvBlocks(in_ch+num_classes, [256, 256, 256], [3, 3, 3], [2, 1, 1])
        self.conv1 = nn.Conv2d(256, 16, kernel_size=(1, 1))  # SE_Block
        self.conv2 = nn.Conv2d(16, out_ch, kernel_size=(1, 1))
        self.SE_maxpool = SE_maxpool
        self.SE_softmax = SE_softmax

    def forward(self, hint_heatmap:torch.Tensor, Fh:torch.Tensor, Fc:torch.Tensor):
        hint = F.interpolate(input=hint_heatmap, size=Fh.size()[2:], mode="bilinear", align_corners=True)
        f = torch.cat((Fh, hint), dim=1)
        f = self.hintEncoder(f)
        if self.SE_maxpool:
            f = f.max(-1)[0].max(-1)[0]
            f = f[:, :, None, None]
        else:
            f = f.mean(-1, keepdim=True).mean(-2, keepdim=True)
        f = self.conv1(f).relu() 
        f = self.conv2(f)
        if self.SE_softmax:  # 得到架構中的A
            f = f.softmax(1)
        else:
            f = f.sigmoid()
        if Fc is not None:
            return f * Fc
        else:
            return f    

class HintFusionLayer(nn.Module):
    def __init__(self, im_ch:int, in_ch:int, out_ch:int=64, ScaleLayer:bool=True):
        """
        im_ch: Num of input image's channels
        in_ch: Num of "hintmap_encoder's input channels", equals to keypoints*2
        out_ch: Num of channels equals to "input channels of HRNet"
        """
        super(HintFusionLayer, self).__init__()
        self.ScaleLayer = ScaleLayer
        init_value = 0.05
        lr_mult = 1
        self.hintmap_encoder = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=16, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=16, out_channels=out_ch, kernel_size=3, stride=2, padding=1)
        )
        self.lr_mult = lr_mult
        self.scale = nn.Parameter(
            torch.full((1,), init_value / lr_mult, dtype=torch.float32)
        )
        self.image_encoder = nn.Sequential(
            nn.Conv2d(in_channels=im_ch, out_channels=out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.last_encoder = nn.Sequential(
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input_image:torch.Tensor, hint_heatmap:torch.Tensor, prev_heatmap:torch.Tensor):
        f = torch.cat((hint_heatmap, prev_heatmap), dim=1)
        f = self.hintmap_encoder(f)
        if self.ScaleLayer:
            scale = torch.abs(self.scale * self.lr_mult)
            f = f * scale
        f = f + self.image_encoder(input_image)
        f = self.last_encoder(f)
        return f



class UNet_IKEM(nn.Module):
    def __init__(self, image_size=(512, 256), num_of_keypoints=68, pretrained_model_path=None, use_iggnet=True):
        super(UNet_IKEM, self).__init__()
        self.unet = UNet(3, num_of_keypoints, )
        