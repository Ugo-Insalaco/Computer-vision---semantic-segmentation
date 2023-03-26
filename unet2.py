import torch.nn as nn
import torch

class Unet2(nn.Module):
    def __init__(self):
        super().__init__()
        # First layer
        self.conv1 = nn.Conv2d(3, 64, 3, padding = 1)
        self.conv2 = nn.Conv2d(64, 16, 3, padding = 1)
        self.batchn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size= (2, 2), stride = 2)
        
        # Second layer
        self.conv3 = nn.Conv2d(16, 64, 3, padding = 1)
        self.conv4 = nn.Conv2d(64, 32, 3, padding = 1)
        self.batchn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size= (2, 2), stride = 2)
        self.drop2 = nn.Dropout(0.5)

        # Third layer
        self.conv5 = nn.Conv2d(32, 32, 3, padding = 1)
        self.conv6 = nn.Conv2d(32, 64, 3, padding = 1)
        self.batchn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size= (2, 2), stride = 2)

        # Fourth layer
        self.conv7 = nn.Conv2d(64, 256, 3, padding = 1)
        self.conv8 = nn.Conv2d(256, 128, 3, padding = 1)
        self.batchn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size= (2, 2), stride = 2)
        self.drop4 = nn.Dropout(0.5)

        # Fifth layer
        self.conv9 = nn.Conv2d(128, 512, 3, padding = 1)
        self.conv10 = nn.Conv2d(512, 256, 3, padding = 1)
        self.batchn5 = nn.BatchNorm2d(256)
        # Upsamling 1
        self.upsample1 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)

        # Sixth layer
        self.conv11 = nn.Conv2d(256, 512, 3, padding = 1)
        self.conv12 = nn.Conv2d(512, 128, 3, padding = 1)
        self.batchn6 = nn.BatchNorm2d(128)
        # Upsamling 2
        self.upsample2 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1,  output_padding=1)

        # Seventh layer
        self.conv13 = nn.Conv2d(128, 256, 3, padding = 1)
        self.conv14 = nn.Conv2d(256, 64, 3, padding = 1)
        self.batchn7 = nn.BatchNorm2d(64)
        # Upsamling 3
        self.upsample3 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)

        # Eighth layer
        self.conv15 = nn.Conv2d(64, 64, 3, padding = 1)
        self.conv16 = nn.Conv2d(64, 32, 3, padding = 1)
        self.batchn8 = nn.BatchNorm2d(32)
        self.upsample4 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1,  output_padding=1)

        # Ninth layer
        self.conv17 = nn.Conv2d(32, 64, 3, padding = 1)
        self.conv18 = nn.Conv2d(64, 16, 3, padding = 1)
        self.batchn9 = nn.BatchNorm2d(16)

        # Last layer
        self.conv19 = nn.Conv2d(16, 1, 1, padding=0)

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        # First layer
        c1 = self.activation(self.batchn1(self.conv2(self.conv1(x))))
        p1 = self.pool1(c1)
        
        # Second layer
        c2 = self.activation(self.batchn2(self.conv4(self.conv3(p1))))
        p2 = self.drop2(self.pool2(c2))

        # Third layer
        c3 = self.activation(self.batchn3(self.conv6(self.conv5(p2))))
        p3 = self.pool3(c3) 

        # Fourth layer
        c4 = self.activation(self.batchn4(self.conv8(self.conv7(p3))))
        p4 = self.drop4(self.pool3(c4))

        # Fifth layer
        c5 = self.activation(self.batchn5(self.conv10(self.conv9(p4))))
        u6 = self.upsample1(c5)
        u6 = torch.cat([u6, c4], dim = 1)
        
        # Sixth layer
        c6 = self.activation(self.batchn6(self.conv12(self.conv11(u6))))
        u7 = self.upsample2(c6)
        u7 = torch.cat([u7, c3], dim = 1)

        # Seventh layer
        c7 = self.activation(self.batchn7(self.conv14(self.conv13(u7))))
        u8 = self.upsample3(c7)
        u8 = torch.cat([u8, c2], dim = 1)

        # Eighth layer
        c8 = self.activation(self.batchn8(self.conv16(self.conv15(u8))))
        u9 = self.upsample4(c8)
        u9 = torch.cat([u9, c1], dim = 1)

        # Ninth layer
        c9 = self.activation(self.batchn9(self.conv18(self.conv17(u9))))

        output = torch.sigmoid((self.conv19(c9)))

        return output