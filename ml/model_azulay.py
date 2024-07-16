import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.padding_size = 1  # Adjust padding to maintain dimensions
        self.kernel_size = 3  # Increase kernel size to 3
        
        # Encoder
        self.conv1 = nn.Conv2d(1, 16, kernel_size=self.kernel_size, padding=self.padding_size, bias=False)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=self.kernel_size, padding=self.padding_size, bias=False)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=self.kernel_size, padding=self.padding_size, bias=False)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=self.kernel_size, padding=self.padding_size, bias=False)
        
        # Decoder
        self.conv5 = nn.Conv2d(128, 64, kernel_size=self.kernel_size, padding=self.padding_size, bias=False)
        self.conv6 = nn.Conv2d(128, 64, kernel_size=self.kernel_size, padding=self.padding_size, bias=False)
        self.conv7 = nn.Conv2d(64, 32, kernel_size=self.kernel_size, padding=self.padding_size, bias=False)
        self.conv8 = nn.Conv2d(64, 32, kernel_size=self.kernel_size, padding=self.padding_size, bias=False)
        self.conv9 = nn.Conv2d(32, 16, kernel_size=self.kernel_size, padding=self.padding_size, bias=False)
        self.conv10 = nn.Conv2d(32, 16, kernel_size=self.kernel_size, padding=self.padding_size, bias=False)
        self.conv11 = nn.Conv2d(16, 1, kernel_size=self.kernel_size, padding=self.padding_size, bias=False)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

        # Extra layers
        self.extra_conv1 = nn.Conv2d(2, 2, kernel_size=self.kernel_size, padding=self.padding_size, bias=False)
        self.extra_conv2 = nn.Conv2d(2, 2, kernel_size=self.kernel_size, padding=self.padding_size, bias=False)
        self.extra_conv3 = nn.Conv2d(2, 2, kernel_size=self.kernel_size, padding=self.padding_size, bias=False)
        self.extra_conv4 = nn.Conv2d(2, 1, kernel_size=self.kernel_size, padding=self.padding_size, bias=False)
        
        # Activation functions
        self.act = nn.ReLU()

    def forward(self, x):
        x = x / (torch.max(x) + 1e-8)

        # Encoder
        x1 = self.act(self.downsample(self.conv1(x)))
        x2 = self.act(self.downsample(self.conv2(x1)))
        x3 = self.act(self.downsample(self.conv3(x2)))
        x4 = self.act(self.downsample(self.conv4(x3)))
        
        # Decoder
        x5 = self.act(self.upsample(self.conv5(x4)))
        x6 = torch.cat([x5, x3], dim=1)
        x6 = self.act(self.conv6(x6))

        x7 = self.act(self.upsample(self.conv7(x6)))
        x8 = torch.cat([x7, x2], dim=1)
        x8 = self.act(self.conv8(x8))

        x9 = self.act(self.upsample(self.conv9(x8)))
        x9 = F.interpolate(x9, size=x1.shape[2:])
        x10 = torch.cat([x9, x1], dim=1)
        x10 = self.act(self.conv10(x10))

        x11 = self.upsample(self.conv11(x10))

        # Extra layers
        x11 = torch.cat([x, x11], dim=1)
        x12 = self.act(self.extra_conv1(x11))
        x13 = self.act(self.extra_conv2(x12))
        x14 = self.act(self.extra_conv3(x13))
        x15 = self.act(self.extra_conv4(x14))

        return x15
