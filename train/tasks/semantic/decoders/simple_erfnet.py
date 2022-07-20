import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class UpsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)

class non_bottleneck_1d (nn.Module):
    def __init__(self, chann, dropprob, dilated):        
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1,0), bias=True)

        self.conv1x3_1 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,1), bias=True)

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1*dilated,0), bias=True, dilation = (dilated,1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,1*dilated), bias=True, dilation = (1, dilated))

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)
        

    def forward(self, input):

        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)

        return F.relu(output+input)    #+input = identity (residual connection)



class Decoder (nn.Module):
    def __init__(self, params, stub_skips, OS, feature_depth=128):
        super(Decoder, self).__init__()

        self.last_channels = 16
        self.backbone_feature_depth = feature_depth
        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlock(self.backbone_feature_depth, 32))
        # self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(32, 0, 1))

        # self.layers.append(UpsamplerBlock(64, 16))
        # self.layers.append(non_bottleneck_1d(16, 0, 1))
        self.layers.append(non_bottleneck_1d(32, 0, 1))

        self.output_conv = nn.ConvTranspose2d(32, 16, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, X, skips):
        output = X
        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)

        return output

    def get_last_depth(self):
      return self.last_channels
