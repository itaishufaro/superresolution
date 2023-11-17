from torch import nn
import torch
from torchvision.models import resnet50

class DenseLayer(nn.Module):
    # A dense layer with dropout and batch normalization with parameter in and out channels
    def __init__(self, in_channels, out_channels, drop_prob=0.1):
        super(DenseLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.drop_prob = drop_prob
        self.features = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1)),
            nn.Dropout2d(self.drop_prob),
        )

    def forward(self, x):
        y = self.features(x)
        concat = torch.cat([x, y], dim=1)
        return concat


class EncodingDenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, drop_prob=0.5):
        self.in_channels = in_channels
        if out_channels is None:
            self.out_channels = self.in_channels
        else:
            self.out_channels = out_channels
        self.drop_prob = drop_prob
        super(EncodingDenseBlock, self).__init__()
        self.features = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(),
            nn.Conv2d(self.in_channels, self.out_channels, (3, 3), (1, 1), (1, 1)),
            nn.Dropout2d(self.drop_prob),
        )

    def forward(self, x):
        y = self.features(x)
        return y


class EncodingConnectionBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, drop_prob=0.5):
        self.in_channels = in_channels
        if out_channels is None:
            self.out_channels = self.in_channels
        else:
            self.out_channels = out_channels
        self.drop_prob = drop_prob
        super(EncodingConnectionBlock, self).__init__()
        self.features = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(),
            nn.Conv2d(self.in_channels, self.out_channels, (1, 1), (1, 1), (0, 0)),
            nn.Dropout2d(self.drop_prob),
            nn.MaxPool2d((2, 2), (2, 2))
        )

    def forward(self, x):
        y = self.features(x)
        return y


class Generator2(nn.Module):
    def __init__(self, drop_prob=0.1, scale_factor=4):
        super(Generator2, self).__init__()
        self.drop_prob = drop_prob
        self.scale_factor = scale_factor
        self.layer1 = nn.Conv2d(1, 32, (3, 3), (1, 1), (1, 1))
        self.layer2 = EncodingDenseBlock(32, 64, drop_prob=self.drop_prob)
        self.layer3 = EncodingConnectionBlock(64, 64, drop_prob=self.drop_prob)

        self.layer4 = EncodingDenseBlock(64, 64, drop_prob=self.drop_prob)
        self.layer5 = EncodingConnectionBlock(64, 64, drop_prob=self.drop_prob)

        self.layer6 = EncodingDenseBlock(64, 64, drop_prob=self.drop_prob)
        self.layer7 = EncodingConnectionBlock(64, 64, drop_prob=self.drop_prob)

        self.layer8 = EncodingDenseBlock(64, 64, drop_prob=self.drop_prob)
        self.layer9 = EncodingConnectionBlock(64, 64, drop_prob=self.drop_prob)

        self.layer10 = EncodingDenseBlock(64, 64, drop_prob=self.drop_prob)
        self.layer11 = nn.ConvTranspose2d(64, 64, (3, 3), (2, 2), (1, 1))

        self.layer12 = EncodingDenseBlock(128, 128, drop_prob=self.drop_prob)
        self.layer13 = nn.ConvTranspose2d(128, 64, (3, 3), (2, 2), (1, 1))

        self.layer14 = EncodingDenseBlock(128, 128, drop_prob=self.drop_prob)
        self.layer15 = nn.ConvTranspose2d(128, 64, (3, 3), (2, 2), (1, 1))

        self.layer16 = EncodingDenseBlock(128, 128, drop_prob=self.drop_prob)
        self.layer17 = nn.ConvTranspose2d(128, 64, (3, 3), (2, 2), (1, 1))
        self.layer18 = EncodingDenseBlock(128, 128, drop_prob=self.drop_prob)
        self.layer19 = nn.ConvTranspose2d(128, 64, (3, 3), (2, 2), (1, 1))
        self.layer20 = EncodingDenseBlock(64, 64, drop_prob=self.drop_prob)
        self.layer21 = nn.ConvTranspose2d(64, 64, (3, 3), (2, 2), (1, 1))
        self.layer22 = nn.Conv2d(64, 1, (3, 3), (1, 1), (1, 1))
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def forward(self, x):
        x = self.layer1(x)
        x1 = self.layer2(x)
        x2 = self.layer4(self.layer3(x1))
        x3 = self.layer6(self.layer5(x2))
        x4 = self.layer8(self.layer7(x3))
        x5 = self.layer11(self.layer10(self.layer9(x4)), output_size=(x4.size(2),x4.size(3)))
        x6 = self.layer13(self.layer12(torch.cat([x4, x5], dim=1)), output_size=(x3.size(2),x3.size(3)))
        x7 = self.layer15(self.layer14(torch.cat([x3, x6], dim=1)), output_size=(x2.size(2),x2.size(3)))
        x8 = self.layer17(self.layer16(torch.cat([x7, x2], dim=1)), output_size=(x1.size(2),x1.size(3)))
        if self.scale_factor == 4:
            x9 = self.layer19(self.layer18(torch.cat([x8, x1], dim=1)), output_size=(x.size(2)*int(self.scale_factor/2),x.size(3)*int(self.scale_factor/2)))
            x10 = self.layer21(self.layer20(x9), output_size=(x.size(2)*self.scale_factor,x.size(3)*self.scale_factor))
        else:
            x10 = self.layer19(self.layer18(torch.cat([x8, x1], dim=1)), output_size=(x.size(2)*self.scale_factor,x.size(3)*self.scale_factor))
        # x9 = self.layer19(self.layer18(torch.cat([x8, x1], dim=1)), output_size=(x.size(2)*int(self.scale_factor/2),x.size(3)*int(self.scale_factor/2)))
        # x10 = self.layer22(self.layer21(self.layer20(x9), output_size=(x.size(2)*self.scale_factor,x.size(3)*self.scale_factor)))
        # x10 = self.layer22(x9)
        x11 = self.sigmoid(self.layer22(x10))
        return x11

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


class ResidualBlock(nn.Module):
    def __init__(self, channel_in, channel_out=None, drop_prob=0.1):
        super (ResidualBlock, self).__init__()
        self.channel_in = channel_in
        if channel_out is None:
            self.channel_out = channel_in
        else:
            self.channel_out = channel_out
        self.drop_prob = drop_prob
        self.features = nn.Sequential(
            nn.Conv2d(self.channel_in, self.channel_out, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(self.channel_out),
            nn.Tanh(),
            nn.Dropout2d(self.drop_prob),
            nn.Conv2d(self.channel_out, self.channel_out, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(self.channel_out)
        )

    def forward(self, x):
        return x + self.features(x)


class EndBlock(nn.Module):
    def __init__(self, channel_in, channel_out=None, up_res=2):
        super (EndBlock, self).__init__()
        self.channel_in = channel_in
        if channel_out is None:
            self.channel_out = channel_in
        else:
            self.channel_out = channel_out
        self.up_res = up_res
        self.features = nn.Sequential(
            nn.Conv2d(self.channel_in, self.channel_out, (3, 3), (1, 1), (1, 1)),
            nn.PixelShuffle(self.up_res),
            nn.Tanh()
        )

    def forward(self, x):
        return self.features(x)


class Generator(nn.Module):
    def __init__(self, n_channels=32, n_channels_end=128, up_res=2, drop_prob=0.1):
        super(Generator, self).__init__()
        self.up_res = up_res
        self.drop_prob = drop_prob
        self.n_channels = n_channels
        self.n_channels_end = n_channels_end
        self.features1 = nn.Sequential(
            nn.Conv2d(1, self.n_channels, (3, 3), (1, 1), (1, 1)),
            nn.Tanh(),
            ResidualBlock(self.n_channels, drop_prob=self.drop_prob),
            ResidualBlock(self.n_channels, drop_prob=self.drop_prob),
            ResidualBlock(self.n_channels, drop_prob=self.drop_prob),
            ResidualBlock(self.n_channels, drop_prob=self.drop_prob),
            ResidualBlock(self.n_channels, drop_prob=self.drop_prob),
            nn.Conv2d(self.n_channels, self.n_channels, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(self.n_channels)
        )
        self.features2 = nn.Sequential(
            EndBlock(self.n_channels, self.n_channels_end, up_res=self.up_res),
            EndBlock(self.n_channels, self.n_channels_end, up_res=self.up_res),
            nn.Conv2d(self.n_channels, 1, (3, 3), (1, 1), (1, 1)),
            nn.Sigmoid()
        )
        self.init_weight()

    def forward(self, x):
        x1 = self.features1(x)
        y = x + x1
        out = self.features2(y)
        return out

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()


class Discriminator(nn.Module):
    def __init__(self, final_size = 1000):
        super(Discriminator, self).__init__()
        self.features = resnet50(weights=None)
        self.final_size = final_size
        self.fc = nn.Linear(self.final_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        rep_x = x.repeat(1, 3, 1, 1)
        return self.sigmoid(self.fc(self.features(rep_x)))

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
            else:
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')

