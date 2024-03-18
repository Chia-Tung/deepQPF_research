from __future__ import annotations

from einops.layers.torch import Rearrange
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn import (
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)

import src.model_architectures as ma
from src.model_architectures.builders.base_builder import BaseBuilder
from src.model_architectures.loss_fn.loss_type import LossType


class TransformerBuilder(BaseBuilder):
    def __init__(
        self,
        model_config,
        loss_config,
        data_info,
        checkpoint_dir,
    ) -> None:
        self._framework = ma.TransformerFramework
        # model config
        self._model_config = self.handle_model_config(model_config)
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
        inp_ch = sum(self._data_info["channel"].values())
        H, W = self._data_info["shape"]
        inp_len = self._data_info["ilen"]
        oup_len = self._data_info["olen"]
        dim_model = self._model_config["d_model"]
        num_head = self._model_config["n_head"]
        num_layers = self._model_config["n_layers"]

        # preprocess
        self.preprocess_layers = nn.ModuleDict(
            {
                "prep_src": self.make_downsample(inp_len, inp_ch, H, W, dim_model),
                "prep_tgt": self.make_downsample(oup_len, 1, H, W, dim_model),
            }
        )

        # encoder
        encode_layer = TransformerEncoderLayer(
            dim_model, num_head, batch_first=True, activation=nn.functional.gelu
        )
        self.encoder = TransformerEncoder(encode_layer, num_layers)

        # decoder
        decode_layer = TransformerDecoderLayer(
            dim_model, num_head, batch_first=True, activation=nn.functional.gelu
        )
        self.decoder = TransformerDecoder(decode_layer, num_layers)

        # post-process
        self.postprocess_layers = self.make_upsample(1, oup_len, H, W, dim_model)

        # loss function
        self.prepare_loss_fn(self._loss_type)

    def prepare_loss_fn(self, loss_type: str) -> TransformerBuilder:
        self._loss_fn = LossType[loss_type].value(self._bal_w, threshold=self._bal_thsh)
        return self

    def make_downsample(self, seq_len, inp_ch, height, width, dmodel):
        msg = "Input gets downsampled by a factor of 6 and 5 sequentially"
        assert height % 30 == 0 and width % 30 == 0, msg
        return nn.Sequential(
            # inp = [B, S, C, 540, 420]
            Rearrange("b s c h w -> (b s) c h w"),
            nn.Conv2d(inp_ch, 2 * inp_ch, kernel_size=6, stride=6),  # [B*S, 2C, 90, 70]
            nn.Conv2d(
                2 * inp_ch, 4 * inp_ch, kernel_size=5, stride=5
            ),  # [B*S, 4C, 18, 14]
            Rearrange("(b s) c h w -> b s (c h w)", s=seq_len),  # [B, S, 4C*18*14]
            nn.Linear(
                4 * inp_ch * (height // 30) * (width // 30), dmodel
            ),  # [B, S, 512]
        )

    def make_upsample(self, oup_ch, oup_len, height, width, dmodel):
        return nn.Sequential(
            nn.Linear(dmodel, 4 * oup_ch * (height // 30) * (width // 30)),
            Rearrange("b s (c h w) -> (b s) c h w", c=4 * oup_ch, h=height // 30),
            nn.ConvTranspose2d(4 * oup_ch, 2 * oup_ch, kernel_size=5, stride=5),
            nn.ConvTranspose2d(2 * oup_ch, 1, kernel_size=6, stride=6),
            Rearrange("(b s) c h w -> b (s c) h w", s=oup_len),
        )

    def build(self) -> LightningModule:
        return self._framework(
            checkpoint_directory=self._checkpoint_dir,
            preprocess=self.preprocess_layers,
            tf_encoder=self.encoder,
            tf_decoder=self.decoder,
            postprocess=self.postprocess_layers,
            loss_fn=self._loss_fn,
            **self._model_config,
        )

    def handle_model_config(self, model_config):
        for k, v in model_config.items():
            if k in ["learning_rate", "adam_epsilon"]:
                model_config[k] = float(v)
        return model_config
