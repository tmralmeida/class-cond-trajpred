from typing import List, Union, Tuple
import torch
import torch.nn as nn
from ..modules import (
    make_mlp,
    get_noise,
    RNNEncoder,
    TFEncoder,
    RNNDecoder,
    MLPDecoder,
)


class Generator(nn.Module):
    def __init__(
        self,
        cfg: dict,
        input_type: Union[str, List[str]],
        visual_feature_cfg: dict | None = None,
        cfg_condition: dict | None = None,
    ) -> None:
        super().__init__()
        # cfg sets
        self.encoder_cfg = cfg["encoder"]
        self.mlp_decoder_ctx_cfg = cfg["mlp_decoder_context"]
        self.decoder_cfg = cfg["decoder"]

        # importatnt atts
        self.noise_type = self.mlp_decoder_ctx_cfg["noise_type"]
        self.noise_dim = self.mlp_decoder_ctx_cfg["noise_dim"]

        if self.encoder_cfg["encoder_type"] == "rnn":
            self.hid_dim = self.encoder_cfg["rnn"]["hidden_dim"]
            self.encoder = RNNEncoder(
                cfg=self.encoder_cfg["rnn"],
                input_type=input_type,
                cfg_condition=cfg_condition,
            )
        elif self.encoder_cfg["encoder_type"] == "tf":
            self.hid_dim = self.encoder_cfg["tf"]["d_model"]
            self.encoder = TFEncoder(cfg=self.encoder_cfg["tf"], input_type=input_type)

        else:
            raise NotImplementedError(self.encoder_cfg["encoder_type"])

        # encoder-decoder connection
        mlp_decoder_noise_dims = [
            self.hid_dim,
            self.hid_dim - self.noise_dim,
        ]
        self.mlp_decoder_context = make_mlp(
            dim_list=mlp_decoder_noise_dims,
            activation=self.mlp_decoder_ctx_cfg["activation"],
            dropout=self.mlp_decoder_ctx_cfg["dropout"],
        )

        # decoder
        if self.decoder_cfg["decoder_type"] == "rnn":
            self.decoder = RNNDecoder(
                cfg=self.decoder_cfg["rnn"],
                visual_feature_cfg=visual_feature_cfg,
                input_type=input_type,
                prediction_len=cfg["prediction_len"],
                cfg_condition=cfg_condition,
            )
        elif self.decoder_cfg["decoder_type"] == "ff":
            self.decoder = MLPDecoder(
                cfg=self.decoder_cfg["ff"],
                visual_feature_cfg=visual_feature_cfg,
                cfg_condition=cfg_condition,
                encoder_hidden_dimension=(
                    self.hid_dim
                    if self.encoder_cfg["encoder_type"] == "rnn"
                    else self.hid_dim * cfg["observation_len"]
                ),
                prediction_len=cfg["prediction_len"],
            )
        else:
            raise NotImplementedError(self.decoder_cfg["decoder_type"])

    def add_noise(
        self, hidden_cell: torch.Tensor
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        hidden_state, cell_state = (
            hidden_cell if isinstance(hidden_cell, tuple) else (hidden_cell, None)
        )

        new_hidden_state = self.mlp_decoder_context(hidden_state)
        if self.encoder_cfg["encoder_type"] == "rnn":
            temp_len, bs, _ = new_hidden_state.shape
        elif self.encoder_cfg["encoder_type"] == "tf":
            bs, temp_len, _ = new_hidden_state.shape
        noise_vector = get_noise((temp_len, self.noise_dim), self.noise_type).to(
            hidden_state
        )

        if self.encoder_cfg["encoder_type"] == "rnn":
            noise_input_decoder = noise_vector.repeat(1, bs, 1)
        elif self.encoder_cfg["encoder_type"] == "tf":
            noise_input_decoder = noise_vector.repeat(bs, 1, 1)
        new_hidden_state = torch.cat([new_hidden_state, noise_input_decoder], dim=-1)
        if cell_state is None:
            return new_hidden_state
        return ((new_hidden_state), (cell_state))

    def forward(self, x: dict, **kwargs):
        encoded_x = self.encoder(x, get_features=False, mask=None, **kwargs)
        noisy_hidden_dim = self.add_noise(encoded_x)
        predictions = self.decoder(
            hidden_cell_state=noisy_hidden_dim, x=x, scaler=kwargs["scaler"]
        )
        return predictions.view(predictions.size(0), -1, 2)


def create_generator(
    model_name: str,
    network_cfg: dict,
    input_type: str | List[str],
    visual_feature_cfg: dict | None = None,
    **kwargs
):
    """returns generator if available"""
    if model_name == "gan":
        generator = Generator(
            cfg=network_cfg,
            input_type=input_type,
            visual_feature_cfg=visual_feature_cfg,
            cfg_condition=None,
        )
    elif model_name == "sup_cgan":
        generator = Generator(
            cfg=network_cfg,
            input_type=input_type,
            visual_feature_cfg=visual_feature_cfg,
            cfg_condition=network_cfg["condition"],
        )
    else:
        raise NotImplementedError(model_name)
    return generator
