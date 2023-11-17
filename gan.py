from torch import nn
import torch
from torchvision.models import resnet50


class DenseLayer(nn.Module):

    """
    DenseLayer is a custom class for a dense layer in a neural network which inherits from PyTorch's Module class.
    It implements a forward pass computation with a dropout layer and Relu activation function.

    Attributes:
        in_channels (int): The number of channels in the input image.
        out_channels (int): The number of output channels produced by the convolution.
        drop_prob (float): the dropout probability for Dropout2D layer.
        features (nn.Sequential): a sequential container.
            Modules will be added to it in the order they are passed in the constructor.
    """

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
        """
        Initialize the EncodingDenseBlock class.

        Args:
        in_channels (int): The number of input channels.
        out_channels (int, optional): The number of output channels. If it is None,
        the number of input channels is used. Default is None.
        drop_prob (float, optional): The dropout probability. Default is 0.5.
        """
        self.in_channels = in_channels
        if out_channels is None:
            self.out_channels = self.in_channels
        else:
            self.out_channels = out_channels
        self.drop_prob = drop_prob
        super(EncodingDenseBlock, self).__init__()

        """
        Create a sequential container.
        Modules will be added to it in the order they are passed in the constructor.
        Alternatively, an ordered dict of modules can also be passed in.
        """
        self.features = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),  # Applies Batch Normalization over a 4D input
            nn.ReLU(),  # Applies the rectified linear unit function element-wise
            nn.Conv2d(self.in_channels, self.out_channels, (3, 3), (1, 1), (1, 1)),
            # Applies a 2D convolution over an input signal composed of several input planes
            nn.Dropout2d(self.drop_prob),
            # Randomly zero out entire channels (a channel is a 2D feature map, e.g.,
            # the j-th channel of the i-th sample in the batched input is a 2D tensor input[i, j]) of the input tensor.
        )

    def forward(self, x):
        y = self.features(x)
        return y


class EncodingConnectionBlock(nn.Module):
    class EncodingConnectionBlock(nn.Module):
        """
        The EncodingConnectionBlock class inherits from the PyTorch nn.Module class.
        This class represents a block of the whole neural network.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int, optional): Number of output channels. If None, it will
            be same as in_channels. Default is None.
            drop_prob (float, optional): Dropout probability. Default is 0.5.
        """
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

        """
        The forward method is implemented to specify what the model has to do at every
        forward step or pass. In this class, it takes input "x" passes it through
        the features and returns the output.

        Arg:
            x (Tensor): A tensor of size (batch_size, in_channels, height, width) where
            height and width are height and width of image respectively.

        Returns:
            y (Tensor): A tensor after being processed by the defined neural network.
        """
        def forward(self, x):
            y = self.features(x)
            return y


class Generator(nn.Module):

    def __init__(self, drop_prob=0.1, scale_factor=4):
        super(Generator, self).__init__()
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
        x5 = self.layer11(self.layer10(self.layer9(x4)), output_size=(x4.size(2), x4.size(3)))
        x6 = self.layer13(self.layer12(torch.cat([x4, x5], dim=1)), output_size=(x3.size(2), x3.size(3)))
        x7 = self.layer15(self.layer14(torch.cat([x3, x6], dim=1)), output_size=(x2.size(2), x2.size(3)))
        x8 = self.layer17(self.layer16(torch.cat([x7, x2], dim=1)), output_size=(x1.size(2), x1.size(3)))
        if self.scale_factor == 4:
            x9 = self.layer19(self.layer18(torch.cat([x8, x1], dim=1)), output_size=(
            x.size(2) * int(self.scale_factor / 2), x.size(3) * int(self.scale_factor / 2)))
            x10 = self.layer21(self.layer20(x9),
                               output_size=(x.size(2) * self.scale_factor, x.size(3) * self.scale_factor))
        else:
            x10 = self.layer19(self.layer18(torch.cat([x8, x1], dim=1)),
                               output_size=(x.size(2) * self.scale_factor, x.size(3) * self.scale_factor))
        x11 = self.sigmoid(self.layer22(x10))
        return x11

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


class Discriminator(nn.Module):

    """
    This is a Discriminator network for a Generative Adversarial Network (GAN).
    The discriminator is a binary classification network that takes an image as input and outputs a scalar prediction
    of 'real' or 'fake'.
    The structure of the Discriminator network is as follows:
    1. Configuration (ResNet-50 Architecture): It uses a residual network (ResNet50) without weights as the feature
    extractor. This network may take as input an image tensor and will output a tensor which represents abstract
    features extracted from the image.
    2. Final Size: The final size of the fully-connected layer of the ResNet-50 is set with `final_size` parameter.
    3. Classification Layer: Following the ResNet-50 architecture, it adds a fully connected linear layer with output
    size 1. This layer is intended to classify the input image as 'real' (from the dataset) or 'fake' (generated).
    4. Sigmoid Layer: It uses a Sigmoid activation function to scale the output of the classification layer to [0, 1],
    where 0 indicates 'fake' and 1 indicates 'real'.
    5. Forward Propagation: In forward propagation, input x is repeated across the color channel
    (if it is not a 3-channel image) then fed into the ResNet-50 architecture and then into the fully connected layer.
    The output of the fully connected layer is then passed through the sigmoid activation function to get the final
    output.
    6. Weight Initialization: The weights of the convolutions and fully connected layers are initialized with He
    initialization (a type of weight initialization that does a good job of setting initial weights in a way that helps
    prevent the problem of training from getting stuck due to poor initialization). Bias terms, if they exist, are
    initialized with zero.
    """
    def __init__(self, final_size=1000):
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
