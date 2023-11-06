import torch
from torch import nn


class SarSubPixel(nn.Module):
    def __init__(self, scale_factor=4, drop_prob=0.1,
                 colors=3):
        '''

        :param scale_factor: How much upscaling we want in our image
        :param drop_prob: How much dropout probability we want
        :param colors: How many color channels does the image have
        '''
        super(SarSubPixel, self).__init__()
        self.scale_factor = scale_factor
        self.drop_prob = drop_prob
        # Feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(colors, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(self.drop_prob)
        )

        # sub-pixel layer
        self.sub_pixel = nn.Sequential(
            nn.Conv2d(32, (scale_factor**2)*colors, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(upscale_factor=self.scale_factor)
        )

    def forward(self, x):
        '''
        :param x: network input image
        :return: network output higher resolution image
        '''
        x = self.features(x)
        x = self.sub_pixel(x)
        return x

    def initialize_weights(self):
        '''
        Initializes all weight according to kaiming normal initialization
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)



class PerceptualLoss(nn.Module):
    def __init__(self, feature_extractor):
        super(PerceptualLoss, self).__init__()
        self.feature_extractor = feature_extractor

    def forward(self, train_hr, target_hr):
        x1 = self.feature_extractor(train_hr)
        x2 = self.feature_extractor(target_hr)
        return nn.MSELoss()(x1, x2)


class SarVAE(nn.Module):
    def __init__(self, scale_factor=4, hidden_dim=128):
        super(SarVAE, self).__init__()
        self.scale_factor = scale_factor
        self.hidden_dim = hidden_dim
        # encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim//4, hidden_dim//2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim//2, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        # decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim//2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim//2, hidden_dim//4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim // 4, 3*(scale_factor**2), kernel_size=4, stride=2, padding=1), # changed stride for super resolution
            nn.PixelShuffle(upscale_factor=self.scale_factor)
        )
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
