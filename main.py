import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchsummary import summary

# since UNET uses a lot of double convolution, create one to be used
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
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
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    #def __init__(self, in_channels=3, out_channels=1, features=[64,128,256,512]):
    def __init__(self, in_channels=3, out_channels=1, features=[32, 64,128 , 256]):
        super(UNET, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # downs
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # ups
        for feature in reversed(features):
            # upsampling
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))

        # bottom middle
        # features[-1] = feature at the end of array
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels,kernel_size=1)

    def forward(self, x):
        skip_connections = []
        # down part of UNET
        for down in self.downs:
            # add each down
            x = down(x)
            # save to concat in up
            skip_connections.append(x)
            # pooling
            x = self.pool(x)
        # bottom middle part of UNET
        x = self.bottleneck(x)
        # reverse the skip connections array since up is the reverse of down
        skip_connections = skip_connections[::-1]

        # up part of UNET
        # in steps of 2 b/c if you remember, up has ConvTranspose and DoubleConv
        for idx in range(0,len(self.ups), 2):
            # add upsampling
            x = self.ups[idx](x)
            # since we are doing it in steps of 2, this gets us elements in skip_connections
            # in order of index 0,1,2,3,... instead of following the step index that the
            # for loop decree of 0,2,4,...
            skip_connection = skip_connections[idx//2]
            # in case size doesn't match
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            # concat
            concat_skip = torch.cat((skip_connection,x), dim=1)
            # add concat
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

def test():
    #x = torch.randn((3, 1, 161, 161))
    model = UNET(in_channels=1, out_channels=1)
    print(model.parameters())
    #preds = model(x)
    #print(preds.shape)
    #print(x.shape)
    #assert preds.shape == x.shape
if __name__ == "__main__":
    test()


