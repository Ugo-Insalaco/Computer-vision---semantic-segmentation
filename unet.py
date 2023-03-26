import torch.nn as nn
import torch
from torchvision.transforms import CenterCrop

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        downChannels = (3, 16, 16, 32, 32, 64, 64, 128, 128, 256, 256)
        upChannels = (256, 128, 128, 64, 64, 32, 32, 16, 16)
        upSamplesChannels = (256, 128, 64, 32, 16)
        finalChannel = 1

        self.downConvs = []
        for i in range(len(downChannels) - 1):
            self.downConvs.append(nn.Conv2d(downChannels[i], downChannels[i+1], (3,3), padding = 1))
        self.maxPool = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.relu = nn.ReLU()

        self.upConvs = []
        for i in range(len(upChannels) - 1):
            self.upConvs.append(nn.Conv2d(upChannels[i], upChannels[i+1], (3,3), padding = 1))

        self.upSamples = []
        for i in range(len(upSamplesChannels) -1):
            self.upSamples.append(nn.ConvTranspose2d(upSamplesChannels[i], upSamplesChannels[i+1], kernel_size=(2,2), stride=2))

        self.finalConv = nn.Conv2d(upChannels[-1], finalChannel, (3,3), padding = 1)
        
    def forward(self, input):
        output = self.downConvs[0](input)
        output = self.downConvs[1](output)
        intermediates = []
        for i in range(2, len(self.downConvs)):
            if(i%2==0):
                intermediates.append(output)
                output = self.maxPool(output)
            output = self.downConvs[i](output)
        
        for i in range(len(self.upSamples)):
            output = self.upSamples[i](output)
            intermed = self.crop(intermediates[-(i+1)], output)
            output = torch.cat([output, intermed], dim=1)
            output = self.upConvs[2*i](output)
            output = self.upConvs[2*i+1](output)

        output = self.finalConv(output)
        return output
    
    def crop(self, encFeatures, x):
        (_, _, H, W) = x.shape
        encFeatures = CenterCrop([H, W])(encFeatures)
        return encFeatures