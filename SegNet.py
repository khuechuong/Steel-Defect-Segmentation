import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchsummary import summary

# since UNET uses a lot of double convolution, create one to be used
class DoubleConvEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConvEncoder, self).__init__()
        self.conv = nn.Sequential(
            # kernel size = 3, stride = 1, pad = 1 (basically same convolution)
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            # normalize in response to bias = false
            nn.BatchNorm2d(out_channels),
            # inplace=True means that it will modify the input directly, without allocating
            # any additional output. It can sometimes slightly decrease the memory usage
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class DoubleConvDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConvDecoder, self).__init__()
        self.conv = nn.Sequential(
            # kernel size = 3, stride = 1, pad = 1 (basically same convolution)
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
            # normalize in response to bias = false
            nn.BatchNorm2d(in_channels),
            # inplace=True means that it will modify the input directly, without allocating
            # any additional output. It can sometimes slightly decrease the memory usage
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class TripleConvDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TripleConvDecoder, self).__init__()
        self.conv = nn.Sequential(
            # kernel size = 3, stride = 1, pad = 1 (basically same convolution)
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
            # normalize in response to bias = false
            nn.BatchNorm2d(in_channels),
            # inplace=True means that it will modify the input directly, without allocating
            # any additional output. It can sometimes slightly decrease the memory usage
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class TripleConvEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TripleConvEncoder, self).__init__()
        self.conv = nn.Sequential(
            # kernel size = 3, stride = 1, pad = 1 (basically same convolution)
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
            # normalize in response to bias = false
            nn.BatchNorm2d(in_channels),
            # inplace=True means that it will modify the input directly, without allocating
            # any additional output. It can sometimes slightly decrease the memory usage
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class SegNet(nn.Module):
    #def __init__(self, in_channels=3, out_channels=1, features=[64,128,256,512]):
    def __init__(self, in_channels=3, out_channels=1, features=[32,64, 128 , 256, 512]):
        super(SegNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # downs
        for feature in features:
            if feature > 64:
                self.downs.append(TripleConvEncoder(in_channels, feature))
                #print(feature)
            else:
                self.downs.append(DoubleConvEncoder(in_channels, feature))
                #print(feature)
            in_channels = feature

        first_512 = True
        # ups
        for feature in reversed(features):
            # upsampling
            self.ups.append(nn.ConvTranspose2d(feature, feature, kernel_size=2, stride=2))
            if feature == 32:
                self.ups.append(DoubleConvDecoder(feature, out_channels))
                #print(out_channels)
            elif feature == 64:
                self.ups.append(DoubleConvDecoder(feature, int(feature/2)))
                #print(feature/2)
            # elif feature == 512:
            #     self.ups.append(TripleConvDecoder(feature, feature))
                #print(feature / 2)
            else:
                self.ups.append(TripleConvDecoder(feature, int(feature / 2)))
                #print(feature)

                # bottom middle
        # features[-1] = feature at the end of array
        self.final_conv = nn.Conv2d(features[0], out_channels,kernel_size=1)

    def forward(self, x):
        skip_connections = []
        # down part of UNET
        for down in self.downs:
            # add each down
            x = down(x)
            #print(f"Conv: {x.size()}")
            # pooling
            x = self.pool(x)
            #print(f"Pool: {x.size()}")
        # up part of UNET
        # in steps of 2 b/c if you remember, up has ConvTranspose and DoubleConv
        #print("Down")
        for up in self.ups:
            # add upsampling
            x = up(x)
            #print(x.size())
        return x

def test():
    x = torch.randn((1, 3, 512, 512))
    model = SegNet(in_channels=3, out_channels=1)
    #print(model.parameters())
    preds = model(x)
    print(f"pred shape: {preds.shape}")
    #print(x.shape)
    #assert preds.shape == x.shape
if __name__ == "__main__":
    test()


