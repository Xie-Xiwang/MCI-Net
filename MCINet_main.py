# -*- coding: utf-8 -*-

#######################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn.functional import softmax
from functools import partial

class EAHead(nn.Module): # External Attention
    def __init__(self, c):# c为上一层的得到的通道数
        super(EAHead, self).__init__()
        # k初始值为32,我认为c表示的是最终将要输出的通道数
        self.k = 128
        self.first_conv = nn.Conv2d(c, c, 1)
        self.k_linear = nn.Conv1d(c, self.k, 1, bias=False)
        self.v_linear = nn.Conv1d(self.k, c, 1, bias=False)

    def forward(self, x):
        idn = x[:]
        b, c, h, w = x.size()
        x = self.first_conv(x)
        x = x.view(b, c, -1)  #
        attn = self.k_linear(x)
        attn = softmax(attn, dim=-1)
        attn = attn / (attn.sum(dim=1, keepdim=True) + 1e-9)
        x = self.v_linear(attn)
        x = x.view(b, c, h, w)
        x = x + idn
        return x


class DoubleConv1(nn.Module):
    def __init__(self, in_ch, out_ch,use_1x1conv=True):
        super(DoubleConv1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            # nn.BatchNorm2d(out_ch),
            # nn.ReLU(inplace=True)
        )
        if use_1x1conv:
            self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1)

    def forward(self, input):
        Y = self.conv(input)
        x = self.conv1(input)
        return F.relu(Y+x)



class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        x=self.conv(input)
        return x
class BR(nn.Module):
    def __init__(self, out_c):
        super(BR, self).__init__()
        # self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)

    def forward(self, x):
        # print('BR输入：',x.shape)
        x_res = self.conv1(x)
        x_res = self.relu(x_res)
        x_res = self.conv2(x_res)

        x = x + x_res

        return x
# 空洞卷积模块
nonlinearity = partial(F.relu, inplace=True)


class DACblock(nn.Module):
    def __init__(self, channel):
        super(DACblock, self).__init__()

        self.dilate2_2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)

        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        # print('DAC输入：',x.shape)
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2_2(self.dilate1(x)))))
        dilate5_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate2_2(self.dilate1(x)))))

        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out+dilate5_out
        return out


# 在每个跳跃连接层加入EA模块
class Mcinet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(Mcinet, self).__init__()

        self.conv1 = DoubleConv1(in_ch, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv1(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv1(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv1(128, 256)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv1(256, 512)

        # center
        self.dblock2 = DACblock(64)
        self.dblock3 = DACblock(128)
        self.dblock4 = DACblock(256)
        self.dblock5 = DACblock(512)
        # self.spp = SPPblock(512)
        self.Att5 = EAHead(512)


        self.br1=BR(64)
        self.br2 = BR(128)
        self.br3 = BR(256)
        self.br4 = BR(512)

        # decoder
        self.conv6 = DoubleConv(512, 256)
        self.conv7 = DoubleConv(256, 128)
        self.conv8 = DoubleConv(128, 64)
        self.conv9 = DoubleConv(64, 32)

        self.up6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.Att4 = EAHead(256)

        self.up7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.Att3 = EAHead(128)

        self.up8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.Att2 = EAHead(64)

        self.up9 = nn.ConvTranspose2d(32, 3, 2, stride=2)
        self.Att1 = EAHead(32)
        self.conv10 = nn.Conv2d(3, out_ch, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)

        # center
        c5 = self.dblock5(c5)
        c5 = self.Att5(c5)
        # c5 = self.spp(c5)


        # decoding + concat path
        up_6 = self.up6(c5)
        A4 = self.Att4(self.dblock4(c4))
        merge6 = torch.cat([A4, up_6], dim=1)
        merge6=self.br4(merge6)
        c6 = self.conv6(merge6)

        up_7 = self.up7(c6)
        A3 = self.Att3(self.dblock3(c3))
        merge7 = torch.cat([A3, up_7], dim=1)
        merge7 = self.br3(merge7)
        c7 = self.conv7(merge7)

        up_8 = self.up8(c7)
        A2 = self.Att2(self.dblock2(c2))
        merge8 = torch.cat([A2, up_8], dim=1)
        merge8 = self.br2(merge8)
        c8 = self.conv8(merge8)

        up_9 = self.br1(c8)
        up_9 = self.conv9(up_9)
        up_9 = self.up9(up_9)

        c10 = self.conv10(up_9)
        out = nn.Sigmoid()(c10)

        return out

