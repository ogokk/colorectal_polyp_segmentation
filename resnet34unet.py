
import torch
import torch.nn as nn
from torchvision import models
from torchsummary import summary as summary_torchsummary

def conv_bn_relu(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

def double_conv_bn_relu(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
    )

class Res34Unet(nn.Module):
    def __init__(self, net_out_ch=1):
        super().__init__()
        encoder = models.resnet34(pretrained=True)
        self.encoder_layers = list(encoder.children())

        self.block1 = nn.Sequential(*self.encoder_layers[:3])
        self.block2 = nn.Sequential(*self.encoder_layers[3:5])
        self.block3 = self.encoder_layers[5]
        self.block4 = self.encoder_layers[6]
        self.block5 = self.encoder_layers[7]

        self.up_6 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.double_6 = double_conv_bn_relu(768, 512)
        self.up_7 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.double_7 = double_conv_bn_relu(384, 256)
        self.up_8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.double_8 = double_conv_bn_relu(192, 128)
        self.up_9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.double_9 = double_conv_bn_relu(128, 32)
        self.up_10 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.conv10 = conv_bn_relu(32, net_out_ch)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)

        x = self.up_6(block5)
        x = torch.cat([x, block4], dim=1)
        x = self.double_6(x)

        x = self.up_7(x)
        x = torch.cat([x, block3], dim=1)
        x = self.double_7(x)

        x = self.up_8(x)
        x = torch.cat([x, block2], dim=1)
        x = self.double_8(x)

        x = self.up_9(x)
        x = torch.cat([x, block1], dim=1)
        x = self.double_9(x)

        x = self.up_10(x)
        x = self.conv10(x)

        return x
    
model = Res34Unet().cuda()
summary_torchsummary(model, (3,448,448))
# Gflops and number of total parameters
# from ptflops import get_model_complexity_info

# with torch.cuda.device(0):
#   # net = models.densenet161()
#   gflops, params = get_model_complexity_info(model, (3, 448, 448), as_strings=True,
#                                             print_per_layer_stat=False, verbose=True)
#   print('{:<30}  {:<8}'.format('Computational complexity: ', gflops))
#   print('{:<30}  {:<8}'.format('Number of parameters: ', params))
