import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        fil_num = 16
        self.conv1 = nn.Conv2d(1, fil_num, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(fil_num, fil_num, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(fil_num, fil_num, kernel_size=(3, 3), padding=1)
        self.conv4 = nn.Conv2d(fil_num, fil_num, kernel_size=(3, 3), padding=1)
        self.conv5 = nn.Conv2d(fil_num, fil_num, kernel_size=(3, 3), padding=1)
        self.conv6 = nn.Conv2d(fil_num, fil_num, kernel_size=(3, 3), padding=1)
        self.conv7 = nn.Conv2d(fil_num, fil_num, kernel_size=(3, 3), padding=1)
        self.conv8 = nn.Conv2d(fil_num, fil_num, kernel_size=(3, 3), padding=1)
        self.conv9 = nn.Conv2d(fil_num, fil_num, kernel_size=(3, 3), padding=1)
        self.conv10 = nn.Conv2d(fil_num, fil_num, kernel_size=(3, 3), padding=1)
        self.conv11 = nn.Conv2d(fil_num, fil_num, kernel_size=(3, 3), padding=1)
        self.conv12 = nn.Conv2d(fil_num, fil_num, kernel_size=(3, 3), padding=1)

        self.reduce_channels = nn.Conv2d(fil_num, 1, kernel_size=(3, 3), padding=1)  # Reduce channels to 1

        self.avgpool = nn.AvgPool2d(kernel_size=(2, 2))

        self.act = nn.ReLU()

    def forward(self, x):
        # normalise input
        x = x / (torch.max(x) + 1e-8)
        x = self.conv1(x)
        la = self.act(self.conv2(x))
        lb = self.act(self.conv3(la))
        la = self.act(self.conv4(lb)) + la
        lb = self.act(self.conv5(la))
        
        apa = self.avgpool(lb)
        apb = self.act(self.conv6(apa))
        apa = self.act(self.conv7(apb)) + apa
        apb = self.act(self.conv8(apa))
        apa = self.act(self.conv9(apb)) + apa
        apb = self.act(self.conv10(apa))
        apa = self.act(self.conv11(apb)) + apa
        apb = self.act(self.conv12(apa))
        apa = self.act(self.conv11(apb)) + apa
        apb = self.act(self.conv12(apa))
        apa = self.act(self.conv11(apb)) + apa

        upa = F.interpolate(apa, scale_factor=2, mode='bicubic') + lb
        upb = self.act(self.conv5(upa))
        upa = self.act(self.conv4(upb)) + upa
        upb = self.act(self.conv3(upa))
        upa = self.act(self.conv2(upb)) + upa
        upb = self.act(self.conv1(self.reduce_channels(upa)))
        upa = self.act(self.conv2(upb)) + upa

        out = self.reduce_channels(upa)

        # set boundary to 0
        # out[:, :, 0, :] = 0
        # out[:, :, -1, :] = 0
        # out[:, :, :, 0] = 0
        # out[:, :, :, -1] = 0

        return out
