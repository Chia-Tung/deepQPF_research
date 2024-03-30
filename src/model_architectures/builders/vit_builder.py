from __future__ import annotations

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from pytorch_lightning import LightningModule
from torchvision.transforms.v2 import CenterCrop, Compose, Normalize, Resize, ToDtype
from transformers import ViTImageProcessor, ViTModel

import src.model_architectures as ma
from src.model_architectures.builders.base_builder import BaseBuilder
from src.model_architectures.loss_fn.loss_type import LossType


class VitBuilder(BaseBuilder):
    def __init__(
        self,
        model_config,
        loss_config,
        data_info,
        checkpoint_dir,
    ) -> None:
        self._framework = ma.VitFramework
        # model config
        self._model_config = self.handle_model_config(model_config)
        self.huggingface_pretrain_name = "google/vit-base-patch16-224-in21k"
        # loss config
        self._loss_type = loss_config["type"]
        self._bal_w = loss_config["balance_weight"]
        self._bal_thsh = loss_config["balance_threshold"]
        # others
        self._data_info = data_info
        self._checkpoint_dir = checkpoint_dir

        # prepare the model components
        self.prepare_components()

    def prepare_components(self):
        processor = ViTImageProcessor.from_pretrained(self.huggingface_pretrain_name)
        size_h = processor.size["height"]
        size_w = processor.size["width"]

        ilen = self._data_info["ilen"]
        shape = self._data_info["shape"]

        self.prepare_img_processor(
            size_h, size_w, ilen, shape
        ).prepare_vit_main_model().prepare_postprocess(
            size_h, size_w, shape
        ).prepare_loss_fn(
            self._loss_type
        )

    def prepare_img_processor(self, size_h, size_w, ilen, shape) -> VitBuilder:
        """
        Convert any matrix of shape (B, S, H, W) to shape (B, S, 224, 224)
        In this case, S=3
        """
        # Not gonna use the default image processor, but the hand-made one
        mean_rain_data = 0  # arbitrarily defined
        std_rain_data = 60  # arbitrarily defined
        self.img_processor = Compose(
            [
                CenterCrop([shape[0], shape[1]]),
                Resize([size_h, size_w]),
                ToDtype(torch.float32, scale=False),
                Normalize(
                    mean=[mean_rain_data for _ in range(ilen)],
                    std=[std_rain_data for _ in range(ilen)],
                ),
            ]
        )
        return self

    def prepare_vit_main_model(self) -> VitBuilder:
        """
        The output of ViT model has 4 types of elements:
            1. last_hidden_state [B, 197, 768], results of the last layer of MLP in ViT
            2. pooler_output [B, 768], results of CLS logit in ViT
            3. hidden_states [13][B, 197, 768], embedding output + 12 MLP outputs
            4. attentions [12][B, 197, 768], 12 self-attn outputs
        """
        self.vit_model = ViTModel.from_pretrained(self.huggingface_pretrain_name)
        self.config = self.vit_model.config
        print(f"[{self.__class__.__name__}] Pretrained Model Config: {self.config}")
        return self

    def prepare_postprocess(self, size_h, size_w, shape) -> VitBuilder:
        """
        Convert (B, 196, 768) to (B, 196, 768) to (B, 3, 224, 224) to (B, 1, 540, 420)
        """
        size_h = int(size_h // self.config.patch_size)
        size_w = int(size_w // self.config.patch_size)
        self.postprocess_layers = nn.Sequential(
            Rearrange("b s d -> b d s"),  # (B, 768, 196)
            Rearrange("b d (h w) -> b d h w", h=size_h, w=size_w),  # (B, 768, 14, 14)
            nn.ConvTranspose2d(
                self.config.hidden_size,
                self.config.num_channels,
                kernel_size=self.config.patch_size,
                stride=self.config.patch_size,
            ),  # (B, 3, 224, 224)
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.ConvTranspose2d(
                self.config.num_channels, 1, kernel_size=2, stride=2
            ),  # (B, 1, 448, 448)
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            Resize([shape[0], shape[1]]),  # (B, 1, 540, 420)
        )
        return self

    def prepare_loss_fn(self, loss_type: str) -> VitBuilder:
        self._loss_fn = LossType[loss_type].value(self._bal_w, threshold=self._bal_thsh)
        return self

    def build(self) -> LightningModule:
        return self._framework(
            # models
            img_processor=self.img_processor,
            vit_model=self.vit_model,
            postprocess=self.postprocess_layers,
            loss_fn=self._loss_fn,
            # checkpoint
            checkpoint_dir=self._checkpoint_dir,
            # scalar
            **self._model_config,
            **self._data_info,
        )
