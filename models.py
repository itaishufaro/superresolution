import torch
from torch import nn
from transformers import pipeline

class SuperResolutionNet(nn.Module):
    def __init__(self, scale_factor=4, drop_prob=0.1):
        super(SuperResolutionNet, self).__init__()
        self.scale_factor = scale_factor
        self.drop_prob = drop_prob
        # Feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(self.drop_prob),
        )

        # Upsampling layers
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(upscale_factor=self.scale_factor)
        )

        self.output = nn.Conv2d(16, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.features(x)
        x = self.upsample(x)
        x = self.output(x)
        return x


class TrainingNetwork(nn.Module):
    def __init__(self, scale_factor=4, drop_prob=0.1):
        super(TrainingNetwork, self).__init__()
        self.scale_factor = scale_factor
        self.drop_prob = drop_prob
        self.superResolution = SuperResolutionNet(scale_factor=self.scale_factor, drop_prob=self.drop_prob)
        resnet = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-1])
        self.resnet_features.requires_grad = False

    def forward(self, train_img, test_img):
        x = self.superResolution(train_img)
        x = self.resnet_features(x)
        y = self.resnet_features(test_img)
        return x, y

    def getsuperres(self):
        return self.superResolution