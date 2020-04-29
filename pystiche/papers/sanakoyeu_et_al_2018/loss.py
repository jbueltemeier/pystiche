from collections import OrderedDict
from typing import Callable, Optional, Sequence, Tuple, Union, Dict

import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy_with_logits

import pystiche
from pystiche.enc import Encoder
from pystiche.loss.perceptual import PerceptualLoss
from pystiche.ops import (
    EncodingOperator,
    EncodingRegularizationOperator,
    MultiLayerEncodingOperator,
)
from pystiche.ops.comparison import MSEEncodingOperator
from pystiche.papers.sanakoyeu_et_al_2018.modules import (
    SanakoyeuEtAl2018DiscriminatorEncoder,
    sanakoyeu_et_al_2018_prediction_module,
    SanakoyeuEtAl2018TransformerBlock,
)

from .utils import ContentOperatorContainer


class DiscriminatorEncodingOperator(EncodingRegularizationOperator):
    def __init__(
        self,
        encoder: Encoder,
        prediction_module: nn.Module,
        score_weight: float = 1e0,
        real: Optional[bool] = True,
    ) -> None:
        super().__init__(encoder, score_weight=score_weight)
        self.pred_module = prediction_module
        self.acc = torch.empty(1)
        self.real = real

    def input_enc_to_repr(
        self, enc: torch.Tensor
    ) -> Union[torch.Tensor, pystiche.TensorStorage]:
        return self.pred_module(enc)

    def process_input_image(
        self, image: torch.Tensor, real: Optional[bool] = None
    ) -> torch.Tensor:
        return self.calculate_score(
            self.input_image_to_repr(image), real if real is not None else self.real
        )

    def _loss(self, prediction: torch.Tensor, real: bool) -> torch.Tensor:
        return binary_cross_entropy_with_logits(
            prediction,
            torch.ones_like(prediction) if real else torch.zeros_like(prediction),
        )

    def _acc(self, prediction: torch.Tensor, real: bool) -> torch.Tensor:
        def get_acc_mask(prediction: torch.Tensor, real: bool):
            if real:
                return torch.masked_fill(
                    torch.zeros_like(prediction),
                    prediction > torch.zeros_like(prediction),
                    1,
                )
            else:
                return torch.masked_fill(
                    torch.zeros_like(prediction),
                    prediction < torch.zeros_like(prediction),
                    1,
                )

        return torch.mean(get_acc_mask(prediction, real))

    def calculate_score(
        self, prediction: torch.Tensor, real: bool = True
    ) -> torch.Tensor:
        self.acc = self._acc(prediction, real)
        return self._loss(prediction, real=real)

    def forward(
        self, input_image: torch.Tensor, real: bool = True
    ) -> Union[torch.Tensor, pystiche.LossDict]:
        return self.process_input_image(input_image, real=real) * self.score_weight

    def get_current_acc(self):
        return self.acc


class MultiLayerDicriminatorEncodingOperator(MultiLayerEncodingOperator):
    def __init__(
        self,
        encoder: SanakoyeuEtAl2018DiscriminatorEncoder,
        layers: Sequence[str],
        get_encoding_op: Callable[[Encoder, float], EncodingOperator],
        layer_weights: Union[str, Sequence[float]] = "sum",
        score_weight: float = 1e0,
    ):
        super().__init__(
            encoder,
            layers,
            get_encoding_op,
            layer_weights=layer_weights,
            score_weight=score_weight,
        )
        self.encoder_parameters = encoder.parameters()

    def get_discriminator_acc(self) -> torch.Tensor:
        acc = []
        for op in self._modules.values():
            if isinstance(op, DiscriminatorEncodingOperator):
                acc.append(op.get_current_acc())
        return torch.mean(torch.stack(acc))

    def parameters(self):
        parameters = list(self.encoder_parameters)
        for op in self.operators():
            if isinstance(op, DiscriminatorEncodingOperator):
                parameters += list(op.pred_module.parameters())
        return parameters

    def process_input_image(
        self, input_image: torch.Tensor, real: Optional[bool] = None
    ) -> pystiche.LossDict:
        return pystiche.LossDict(
            [(name, op(input_image, real)) for name, op in self.named_children()]
        )

    def forward(
        self, input_image: torch.Tensor, real: Optional[bool] = None
    ) -> Tuple[Union[torch.Tensor, pystiche.LossDict], torch.Tensor]:
        return self.process_input_image(input_image, real) * self.score_weight


def sanakoyeu_et_al_2018_discriminator_operator(
    in_channels: int = 3,
    impl_params: bool = True,
    encoder: Optional[SanakoyeuEtAl2018DiscriminatorEncoder] = None,
    layers: Optional[Sequence[str]] = None,
    prediction_modules: Optional[Dict[str, nn.Module]] = None,
    layer_weights: Union[str, Sequence[float]] = "sum",
    score_weight: float = None,
) -> MultiLayerDicriminatorEncodingOperator:

    if encoder is None:
        encoder = SanakoyeuEtAl2018DiscriminatorEncoder(in_channels=in_channels)

    if score_weight is None:
        if impl_params:
            score_weight = 1e0
        else:
            score_weight = 1e-3

    if layers is None:
        layers = ("lrelu0", "lrelu1", "lrelu3", "lrelu5", "lrelu6")

    if prediction_modules is None:
        prediction_modules = {
            "lrelu0": sanakoyeu_et_al_2018_prediction_module(128, 5),
            "lrelu1": sanakoyeu_et_al_2018_prediction_module(128, 10),
            "lrelu3": sanakoyeu_et_al_2018_prediction_module(512, 10),
            "lrelu5": sanakoyeu_et_al_2018_prediction_module(1024, 6),
            "lrelu6": sanakoyeu_et_al_2018_prediction_module(1024, 3),
        }

    assert tuple(prediction_modules.keys()) == layers, (
        "The keys in prediction_modules should match "
        "the entries in layers. However layers "
        + str(layers)
        + " and keys: "
        + str(tuple(prediction_modules.keys()))
        + " are given. "
    )

    def get_encoding_op(encoder, layer_weight):
        return DiscriminatorEncodingOperator(
            encoder, prediction_modules[encoder.layer], score_weight=layer_weight,
        )

    return MultiLayerDicriminatorEncodingOperator(
        encoder,
        layers,
        get_encoding_op,
        layer_weights=layer_weights,
        score_weight=score_weight,
    )


def sanakoyeu_et_al_2018_discriminator_loss(
    discriminator: MultiLayerDicriminatorEncodingOperator,
    output_photo: torch.Tensor,
    input_painting: torch.Tensor,
    input_photo: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:

    loss = discriminator(input_painting, real=True)
    acc = discriminator.get_discriminator_acc()
    for key, value in zip(
        loss.keys(), discriminator(output_photo, real=False).values()
    ):
        loss[key] = loss[key] + value

    acc += discriminator.get_discriminator_acc()
    if input_photo is not None:
        for key, value in zip(
            loss.keys(), discriminator(input_photo, real=False).values()
        ):
            loss[key] = loss[key] + value
        acc += discriminator.get_discriminator_acc()
        return loss, acc / 3
    return loss, acc / 2


class SanakoyeuEtAl2018DiscriminatorLoss(nn.Module):
    def __init__(self, discriminator: MultiLayerDicriminatorEncodingOperator) -> None:
        super().__init__()
        self.discriminator = discriminator
        self.acc = 0.0

    def get_current_acc(self):
        return self.acc

    def forward(
        self,
        output_photo: torch.Tensor,
        input_painting: torch.Tensor,
        input_photo: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss, self.acc = sanakoyeu_et_al_2018_discriminator_loss(
            self.discriminator, output_photo, input_painting, input_photo
        )
        return loss


class SanakoyeuEtAl2018FeatureOperator(MSEEncodingOperator):
    def __init__(
        self, encoder: Encoder, score_weight: float = 1.0, impl_params: bool = True
    ):
        super().__init__(encoder, score_weight=score_weight)
        self.impl_params = impl_params

    def calculate_score(
        self, input_repr: torch.Tensor, target_repr: torch.Tensor, ctx: None
    ) -> torch.Tensor:
        if self.impl_params:
            return torch.mean(torch.abs(input_repr - target_repr))
        else:
            return super().calculate_score(input_repr, target_repr, ctx)


def sanakoyeu_et_al_2018_style_aware_content_loss(
    encoder: Optional[pystiche.SequentialModule],
    impl_params: bool = True,
    score_weight=None,
) -> SanakoyeuEtAl2018FeatureOperator:
    if score_weight is None:
        if impl_params:
            score_weight = 1e2
        else:
            score_weight = 1e0

    return SanakoyeuEtAl2018FeatureOperator(
        encoder, score_weight=score_weight, impl_params=impl_params
    )


def sanakoyeu_et_al_2018_transformed_image_loss(
    transformer_block: Optional[SanakoyeuEtAl2018TransformerBlock] = None,
    impl_params: bool = True,
    score_weight=None,
) -> MSEEncodingOperator:
    if score_weight is None:
        if impl_params:
            score_weight = 1e2
        else:
            score_weight = 1e0

    if transformer_block is None:
        transformer_block = SanakoyeuEtAl2018TransformerBlock()

    return MSEEncodingOperator(transformer_block, score_weight=score_weight)


def sanakoyeu_et_al_2018_transformer_loss(
    encoder: Optional[pystiche.SequentialModule],
    impl_params: bool = True,
    style_aware_content_loss: Optional[SanakoyeuEtAl2018FeatureOperator] = None,
    transformed_image_loss: Optional[MSEEncodingOperator] = None,
    style_loss: Optional[MultiLayerDicriminatorEncodingOperator] = None,
) -> PerceptualLoss:

    if style_aware_content_loss is None:
        style_aware_content_loss = sanakoyeu_et_al_2018_style_aware_content_loss(
            encoder, impl_params=impl_params
        )

    if transformed_image_loss is None:
        transformed_image_loss = sanakoyeu_et_al_2018_transformed_image_loss(
            impl_params=impl_params
        )

    content_loss = ContentOperatorContainer(
        OrderedDict(
            (
                ("style_aware_content_loss", style_aware_content_loss),
                ("tranformed_image_loss", transformed_image_loss),
            )
        )
    )

    if style_loss is None:
        style_loss = sanakoyeu_et_al_2018_discriminator_operator()

    return PerceptualLoss(content_loss, style_loss)
