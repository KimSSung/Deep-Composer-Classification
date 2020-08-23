# ResNet
# https://github.com/FrancescoSaverioZuppichini/ResNet/blob/master/ResNet.ipynb

import torch
import torch.nn as nn

from functools import partial
from dataclasses import dataclass
from collections import OrderedDict

k = (3,3)

def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(
        OrderedDict(
            {
                "conv": conv(in_channels, out_channels, *args, **kwargs),
                "bn": nn.BatchNorm2d(out_channels),
            }
        )
    )

# print(conv_bn(3, 3, nn.Conv2d, kernel_size=k))
# print('---------------------------------------------')



class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (
            self.kernel_size[0] // 2,
            self.kernel_size[1] // 2,
        )  # dynamic add padding based on the kernel_size



conv_x_ = partial(Conv2dAuto, kernel_size=k, bias=False)
conv = conv_x_(in_channels=32, out_channels=64)


# print(conv)
# print('---------------------------------------------')
del conv


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.blocks = nn.Identity()
        self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        if self.should_apply_shortcut:
            residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        return x

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels

class ResNetResidualBlock(ResidualBlock):
    def __init__(
        self,
        in_channels,
        out_channels,
        expansion=1,
        downsampling=1,
        conv=conv_x_,
        *args,
        **kwargs
    ):
        super().__init__(in_channels, out_channels)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
        self.shortcut = (
            nn.Sequential(
                OrderedDict(
                    {
                        "conv": nn.Conv2d(
                            self.in_channels,
                            self.expanded_channels,
                            kernel_size=1,
                            stride=self.downsampling,
                            bias=False,
                        ),
                        "bn": nn.BatchNorm2d(self.expanded_channels),
                    }
                )
            )
            if self.should_apply_shortcut
            else None
        )

    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels


# print(ResNetResidualBlock(32, 64))
# print('---------------------------------------------')


class ResNetBasicBlock(ResNetResidualBlock):
    expansion = 1

    def __init__(self, in_channels, out_channels, activation=nn.ReLU, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(
                self.in_channels,
                self.out_channels,
                conv=self.conv,
                bias=False,
                stride=self.downsampling,
                kernel_size=k

            ),
            activation(),
            nn.Dropout(p=0.25),
            
            conv_bn(
                self.out_channels,
                self.expanded_channels,
                conv=self.conv,
                bias=False,
                kernel_size=k
            ),
            nn.Dropout(p=0.25),
        )

# dummy = torch.ones((1, 32, 224, 224))

# block = ResNetBasicBlock(32, 64)
# block(dummy).shape
# print(block)
# print('---------------------------------------------')



class ResNetBottleNeckBlock(ResNetResidualBlock):
    expansion = 4

    def __init__(self, in_channels, out_channels, activation=nn.ReLU, *args, **kwargs):
        super().__init__(in_channels, out_channels, expansion=4, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(
                self.in_channels, self.out_channels, self.conv, kernel_size=1
            ),
            activation(),
            nn.Dropout(p=0.25),

            conv_bn(
                self.out_channels,
                self.out_channels,
                self.conv,
                stride=self.downsampling,
                kernel_size=k,
            ),
            activation(),
            nn.Dropout(p=0.25),

            conv_bn(
                self.out_channels, self.expanded_channels, self.conv, kernel_size=1
            ),
            nn.Dropout(p=0.25),
        )


# dummy = torch.ones((1, 32, 10, 10))

# block = ResNetBottleNeckBlock(32, 64)
# block(dummy).shape
# print(block)
# print('---------------------------------------------')



class ResNetLayer(nn.Module):
    def __init__(
        self, in_channels, out_channels, block=ResNetBasicBlock, n=1, *args, **kwargs
    ):
        super().__init__()
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        downsampling = 2 if in_channels != out_channels else 1

        self.blocks = nn.Sequential(
            block(
                in_channels, out_channels, *args, **kwargs, downsampling=downsampling
            ),
            *[
                block(
                    out_channels * block.expansion,
                    out_channels,
                    downsampling=1,
                    *args,
                    **kwargs
                )
                for _ in range(n - 1)
            ]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x


dummy = torch.ones((1, 32, 48, 48))

layer = ResNetLayer(64, 128, block=ResNetBasicBlock, n=3)
# layer(dummy).shape
# print(layer)
# print('---------------------------------------------')


class ResNetEncoder(nn.Module):
    """
    ResNet encoder composed by increasing different layers with increasing features.
    """

    def __init__(
        self,
        in_channels=128,
        blocks_sizes=[64, 128, 256, 512],
        deepths=[2, 2, 2, 2],
        activation=nn.ReLU,
        block=ResNetBasicBlock,
        *args,
        **kwargs
    ):
        super().__init__()

        self.blocks_sizes = blocks_sizes

        self.gate = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.blocks_sizes[0],
                kernel_size=7, # all resnet codes write 7 here... so fixed it
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm2d(self.blocks_sizes[0]),
            activation(),
            nn.MaxPool2d(kernel_size=k, stride=2, padding=1),
        )

        # resnet part
        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        self.blocks = nn.ModuleList(
            [
                ResNetLayer(
                    blocks_sizes[0],
                    blocks_sizes[0],
                    n=deepths[0],
                    activation=activation,
                    block=block,
                    *args,
                    **kwargs
                ),
                *[
                    ResNetLayer(
                        in_channels * block.expansion,
                        out_channels,
                        n=n,
                        activation=activation,
                        block=block,
                        *args,
                        **kwargs
                    )
                    for (in_channels, out_channels), n in zip(
                        self.in_out_block_sizes, deepths[1:]
                    )
                ],
            ]
        )

    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x


class ResnetDecoder(nn.Module):
    """
    This class represents the tail of ResNet. It performs a global pooling and maps the output to the
    correct class by using a fully connected layer.
    """

    def __init__(self, in_features, n_classes):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.decoder = nn.Linear(in_features, n_classes)

    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return x


class ResNet(nn.Module):
    def __init__(self, in_channels, n_classes, *args, **kwargs):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels, *args, **kwargs)
        self.decoder = ResnetDecoder(
            self.encoder.blocks[-1].blocks[-1].expanded_channels, n_classes
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def resnet18(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block=ResNetBasicBlock, deepths=[2, 2, 2, 2])


def resnet34(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block=ResNetBasicBlock, deepths=[3, 4, 6, 3])


def resnet50(in_channels, n_classes):
    return ResNet(
        in_channels, n_classes, block=ResNetBottleNeckBlock, deepths=[3, 4, 6, 3]
    )


def resnet101(in_channels, n_classes):
    return ResNet(
        in_channels, n_classes, block=ResNetBottleNeckBlock, deepths=[3, 4, 23, 3]
    )


def resnet152(in_channels, n_classes):
    return ResNet(
        in_channels, n_classes, block=ResNetBottleNeckBlock, deepths=[3, 8, 36, 3]
    )
