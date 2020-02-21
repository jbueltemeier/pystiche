from typing import Union, Optional, Sequence
from torch import nn
from pystiche.enc import MultiLayerEncoder
from pystiche.ops import MSEEncodingOperator, GramOperator, MultiLayerEncodingOperator
from .utils import (
    ulyanov_et_al_2016_multi_layer_encoder,
    ulyanov_et_al_2016_conv_sequence,
    UlaynovEtAl2016LevelBlock,
)


def ulyanov_et_al_2016_transformer(levels=4):
    level_block = ulyanov_et_al_2016_conv_sequence(3, 8)
    for _ in range(levels):
        level_block = UlaynovEtAl2016LevelBlock(
            level_block, ulyanov_et_al_2016_conv_sequence(3, 8)
        )

    modules = (
        level_block,
        nn.Conv2d(40, 3, 1, stride=1),
    )
    return nn.Sequential(*modules)


def ulyanov_et_al_2016_content_loss(
    impl_params: bool = True,
    multi_layer_encoder: Optional[MultiLayerEncoder] = None,
    layer: str = "relu_4_2",
    score_weight=1e0,  # FIXME
):
    # FIXME impl_params?
    # FIXME: loss_reduction?
    if multi_layer_encoder is None:
        multi_layer_encoder = ulyanov_et_al_2016_multi_layer_encoder()
    encoder = multi_layer_encoder[layer]

    return MSEEncodingOperator(encoder, score_weight=score_weight)


def ulyanov_et_al_2016_style_loss(
    impl_params: bool = True,
    multi_layer_encoder: Optional[MultiLayerEncoder] = None,
    layers: Optional[Sequence[str]] = None,
    layer_weights: Union[str, Sequence[float]] = "sum",
    score_weight: float = 1e0,  # FIXME
    **gram_op_kwargs,
):
    # FIXME impl_params?
    # FIXME: normalize?
    # FIXME: loss_reduction?
    if multi_layer_encoder is None:
        multi_layer_encoder = ulyanov_et_al_2016_multi_layer_encoder()

    if layers is None:
        layers = ("relu_1_1", "relu_2_1", "relu_3_1", "relu_4_1")  # FIXME

    def get_encoding_op(encoder, layer_weight):
        return GramOperator(encoder, score_weight=layer_weight, **gram_op_kwargs)

    return MultiLayerEncodingOperator(
        multi_layer_encoder,
        layers,
        get_encoding_op,
        layer_weights=layer_weights,
        score_weight=score_weight,
    )


def ulyanov_et_al_2016_regularization(
    score_weight: float = 1e-6, **total_variation_op_kwargs: Any
):
    # FIXME: see johnson
    return TotalVariationOperator(
        score_weight=score_weight, **total_variation_op_kwargs
    )
