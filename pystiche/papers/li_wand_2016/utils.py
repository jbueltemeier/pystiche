import torch
from torch import optim
from torch.optim.optimizer import Optimizer

from pystiche.enc import MultiLayerEncoder, vgg19_multi_layer_encoder
from pystiche.image import CaffePostprocessing, CaffePreprocessing

__all__ = [
    "li_wand_2016_preprocessor",
    "li_wand_2016_postprocessor",
    "li_wand_2016_multi_layer_encoder",
    "li_wand_2016_optimizer",
]


def li_wand_2016_preprocessor() -> CaffePreprocessing:
    return CaffePreprocessing()


def li_wand_2016_postprocessor() -> CaffePostprocessing:
    return CaffePostprocessing()


def li_wand_2016_multi_layer_encoder() -> MultiLayerEncoder:
    return vgg19_multi_layer_encoder(
        weights="caffe", preprocessing=False, allow_inplace=True
    )


def li_wand_2016_optimizer(input_image: torch.Tensor) -> Optimizer:
    return optim.LBFGS([input_image.requires_grad_(True)], lr=1.0, max_iter=1)
