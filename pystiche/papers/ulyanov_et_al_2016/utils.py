from typing import Union, Tuple, Collection
import torch
from torch import nn, optim
from torch.optim.optimizer import Optimizer
from pystiche.image import CaffePreprocessing, CaffePostprocessing

from pystiche.enc import MultiLayerEncoder, vgg19_encoder


def ulyanov_et_al_2016_preprocessor() -> CaffePreprocessing:
    return CaffePreprocessing()


def ulyanov_et_al_2016_postprocessor() -> CaffePostprocessing:
    return CaffePostprocessing()


def ulyanov_et_al_2016_multi_layer_encoder() -> MultiLayerEncoder:
    return vgg19_encoder(weights="caffe", allow_inplace=True)


def ulyanov_et_al_2016_optimizer(transformer: nn.Module) -> Optimizer:
    return optim.Adam(transformer.parameters(), lr=1e-3)


def get_norm_module(out_channels: int, instance_norm: bool) -> nn.Module:
    if instance_norm:
        return nn.InstanceNorm2d(out_channels)
    else:
        return nn.BatchNorm2d(out_channels)


class UlyanovEtAl2016ConvBlock(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Tuple[int, int], int],
        impl_params: bool = True,
        stride: Union[Tuple[int, int], int] = 1,
        instance_norm: bool = True,
        inplace: bool = True,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels

        def elementwise(fn, inputs):
            if isinstance(inputs, Collection):
                return tuple([fn(input) for input in inputs])
            return fn(inputs)

        def same_size_padding(kernel_size):
            return elementwise(lambda x: (x - 1) // 2, kernel_size)

        padding = same_size_padding(kernel_size)

        modules = []

        if padding > 0:
            modules.append(nn.ReflectionPad2d(padding))

        modules.append(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=0,)
        )

        modules.append(get_norm_module(out_channels, instance_norm))

        if impl_params:
            activation = nn.ReLU(inplace=inplace)
        else:
            activation = nn.LeakyReLU(negative_slope=1e8, inplace=inplace)
        modules.append(activation)

        super().__init__(*modules)


# def ulyanov_et_al_2016_conv_block(
#     in_channels: int,
#     out_channels: int,
#     impl_params: bool = True,
#     kernel_size: Union[Tuple[int, int], int] = 3,
#     stride: Union[Tuple[int, int], int] = 1,
#     instance_norm: bool = True,
#     inplace: bool = True,
# ) -> UlyanovEtAl2016ConvBlock:
#     return UlyanovEtAl2016ConvBlock(
#         in_channels,
#         out_channels,
#         impl_params=impl_params,
#         kernel_size=kernel_size,
#         stride=stride,
#         instance_norm=instance_norm,
#         inplace=inplace,
#     )


class UlyanovEtAl2016ConvSequence(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        impl_params: bool = True,
        instance_norm: bool = True,
        inplace: bool = True,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels

        modules = (
            UlyanovEtAl2016ConvBlock(
                in_channels,
                out_channels,
                3,
                impl_params=impl_params,
                instance_norm=instance_norm,
                inplace=inplace,
            ),
            UlyanovEtAl2016ConvBlock(
                out_channels,
                out_channels,
                3,
                impl_params=impl_params,
                instance_norm=instance_norm,
                inplace=inplace,
            ),
            UlyanovEtAl2016ConvBlock(
                out_channels,
                out_channels,
                1,
                impl_params=impl_params,
                instance_norm=instance_norm,
                inplace=inplace,
            ),
        )

        super().__init__(*modules)


def ulyanov_et_al_2016_conv_sequence(
    in_channels: int,
    out_channels: int,
    impl_params: bool = True,
    instance_norm: bool = True,
    inplace: bool = True,
) -> UlyanovEtAl2016ConvSequence:
    return UlyanovEtAl2016ConvSequence(
        in_channels,
        out_channels,
        impl_params=impl_params,
        instance_norm=instance_norm,
        inplace=inplace,
    )


class UlaynovEtAl2016JoinBlock(nn.Module):
    def __init__(
        self,
        deep_block: UlyanovEtAl2016ConvSequence,
        shallow_block: UlyanovEtAl2016ConvSequence,
        instance_norm: bool = True,
    ) -> None:
        super().__init__()
        self.deep_block = deep_block
        self.deep_norm = get_norm_module(deep_block.out_channels, instance_norm)
        self.deep_upsample = nn.Upsample(scale_factor=2.0, mode="nearest")

        self.shallow_block = shallow_block
        self.shallow_norm = get_norm_module(shallow_block.out_channels, instance_norm)

        self.out_channels = self.deep_block.out_channels + shallow_block.out_channels

    def forward(
        self, deep_input: torch.Tensor, shallow_input: torch.Tensor
    ) -> torch.Tensor:
        deep_output = self.deep_norm(self.deep_upsample(self.deep_block(deep_input)))
        shallow_output = self.shallow_norm(self.shallow_block(shallow_input))
        return torch.cat((deep_output, shallow_output), dim=1)


class UlaynovEtAl2016LevelBlock(nn.Module):
    def __init__(
        self,
        deep_block: Union[UlyanovEtAl2016ConvSequence, "UlaynovEtAl2016LevelBlock"],
        shallow_block: UlyanovEtAl2016ConvSequence,
        impl_params: bool = True,
        instance_norm: bool = True,
        inplace: bool = True,
    ):
        super().__init__()
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.join = UlaynovEtAl2016JoinBlock(deep_block, shallow_block)
        out_channels = self.join.out_channels
        self.conv_sequence = UlyanovEtAl2016ConvSequence(
            out_channels,
            out_channels,
            impl_params=impl_params,
            instance_norm=instance_norm,
            inplace=inplace,
        )
        self.out_channels = out_channels

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        deep_input = self.downsample(input)
        shallow_input = input
        join_output = self.join(deep_input, shallow_input)
        return self.conv_sequence(join_output)
