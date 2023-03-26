import torch.nn as nn
import torch

class ConvLayer(nn.Module):
    def __init__(self, channels, kernel_size, padding):
        super().__init__()
        self.convs = []
        for i in range(1, len(channels)):
            self.convs.append(nn.Conv2d(channels[i-1], channels[i], kernel_size, padding=padding))
        self.batchn = nn.BatchNorm2d(channels[-1])
        self.activation = nn.ReLU(inplace=True)
        
    def forward(self, x):
        output = x
        for layer in self.convs:
            output = layer(output)
        output = self.batchn(output)
        output = self.activation(output)
        return output
    
class Unet3(nn.Module):
    def __init__(self):
        super().__init__()
        # First layer
        self.conv1 = ConvLayer((3, 64, 16), 3, 1)
        self.pool1 = nn.MaxPool2d(kernel_size= (2, 2), stride = 2)
        
        # Second layer
        self.conv2 = ConvLayer((16, 64, 32), 3, 1)
        self.pool2 = nn.MaxPool2d(kernel_size= (2, 2), stride = 2)
        self.drop2 = nn.Dropout(0.5)

        # Third layer
        self.conv3 = ConvLayer((32, 32, 64), 3, 1)
        self.pool3 = nn.MaxPool2d(kernel_size= (2, 2), stride = 2)

        # Fourth layer
        self.conv4 = ConvLayer((64, 256, 128), 3, 1)
        self.pool4 = nn.MaxPool2d(kernel_size= (2, 2), stride = 2)
        self.drop4 = nn.Dropout(0.5)

        # Fifth layer
        self.conv5 = ConvLayer((128, 512, 256), 3, 1)
        # Upsamling 1
        self.upsample1 = nn.ConvTranspose2d(256, 128, kernel_size=(2,2), stride=2)

        # Sixth layer
        self.conv6 = ConvLayer((256, 512, 128), 3, 1)
        # Upsamling 2
        self.upsample2 = nn.ConvTranspose2d(128, 64, kernel_size=(2,2), stride=2)

        # Seventh layer
        self.conv7 = ConvLayer((128, 256, 64), 3, 1)
        # Upsamling 3
        self.upsample3 = nn.ConvTranspose2d(64, 32, kernel_size=(2,2), stride=2)

        # Eighth layer
        self.conv8 = ConvLayer((64, 64, 32), 3, 1)
        # Upsamling 3
        self.upsample4 = nn.ConvTranspose2d(32, 16, kernel_size=(2,2), stride=2)

        # Ninth layer
        self.conv9 = ConvLayer((32, 64, 16), 3, 1)

        # Last layer
        self.lastConv = nn.Conv2d(16, 1, 1, padding=0)

    def forward(self, x):
        # First layer
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        
        # Second layer
        c2 = self.conv2(p1)
        p2 = self.drop2(self.pool2(c2))

        # Third layer
        c3 = self.conv3(p2)
        p3 = self.pool3(c3) 

        # Fourth layer
        c4 = self.conv4(p3)
        p4 = self.drop4(self.pool4(c4))

        # Fifth layer
        c5 = self.conv5(p4)
        u6 = self.upsample1(c5)
        u6 = torch.cat([u6, c4], dim = 1)
        
        # Sixth layer
        c6 = self.conv6(u6)
        u7 = self.upsample2(c6)
        u7 = torch.cat([u7, c3], dim = 1)

        # Seventh layer
        c7 = self.conv7(u7)
        u8 = self.upsample3(c7)
        u8 = torch.cat([u8, c2], dim = 1)

        # Eighth layer
        c8 = self.conv8(u8)
        u9 = self.upsample4(c8)
        u9 = torch.cat([u9, c1], dim = 1)

        # Ninth layer
        c9 = self.conv9(u9)
        
        output = torch.sigmoid((self.lastConv(c9)))

        return output