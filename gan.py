from torch import nn
import torch
from torchvision.models import resnet50, ResNet50_Weights

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

#
#
# class TransitionDown(nn.Module):
#     def __init__(self, in_channels, out_channels=None, drop_prob=0.2):
#         '''
#         FC-DenseNet Transition Down module, with dropout and batch normalization
#         :param in_channels:
#         :param out_channels:
#         '''
#         super(TransitionDown, self).__init__()
#         self.in_channels = in_channels
#         self.drop_prob = drop_prob
#         if out_channels is None:
#             self.out_channels = in_channels
#         else:
#             self.out_channels = out_channels
#
#         self.features = nn.Sequential(
#             nn.BatchNorm2d(self.in_channels),
#             nn.ReLU(),
#             nn.Conv2d(self.in_channels, self.out_channels, (1, 1), (1, 1)),
#             nn.Dropout2d(self.drop_prob),
#             nn.MaxPool2d((2, 2), stride=2),
#         )
#
#     def forward(self, x):
#         return self.features(x)
#
#
# class TransitionUp(nn.Module):
#     def __init__(self, in_channels, out_channels=None, drop_prob=0.2):
#         '''
#         FC-DenseNet Transition Up module, with dropout and batch normalization
#         :param in_channels:
#         :param out_channels:
#         '''
#         super(TransitionUp, self).__init__()
#         self.in_channels = in_channels
#         self.drop_prob = drop_prob
#         if out_channels is None:
#             self.out_channels = in_channels
#         else:
#             self.out_channels = out_channels
#
#         self.features = nn.Sequential(
#             nn.ConvTranspose2d(self.in_channels, self.out_channels, (3, 3), (2, 2), (1, 1), (1, 1))
#         )
#
#     def forward(self, x):
#         return self.features(x)
#
#
# class DenseBlock(nn.Module):
#     def __init__(self, n_layers, in_channels, growth_rate=16, keep_input=True):
#         super(DenseBlock, self).__init__()
#
#         self.n_layers = n_layers
#         self.in_channels = in_channels
#         self.growth_rate = growth_rate
#         self.keep_input = keep_input
#
#         if self.keep_input:
#             self.out_channels = in_channels + n_layers * growth_rate
#         else:
#             self.in_channels = n_layers * growth_rate
#
#         self.block = self._build_block()
#
#     def _build_block(self):
#         in_channels = self.in_channels
#         layers = []
#         for i in range(self.n_layers):
#             layers.append(DenseLayer(in_channels, self.growth_rate))
#             in_channels += self.growth_rate
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         out = self.block(x)
#         if not self.keep_input:
#             out = out[:, -self.in_channels:, ...]
#         return out


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


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, drop_prob=0.1,
                 kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(DiscriminatorBlock, self).__init__()
        self.in_channels = in_channels
        if out_channels is None:
            self.out_channels = in_channels
        else:
            self.out_channels = out_channels
        self.drop_prob = drop_prob
        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(self.out_channels),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.features(x)


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

