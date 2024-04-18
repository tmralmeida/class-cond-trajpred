from typing import List, Tuple
import torch
import torch.nn as nn
from ..modules import (
    make_mlp,
    RNNEncoder,
    TFEncoder,
    RNNDecoder,
    MLPDecoder,
)


class VariationalAutoEncoder(nn.Module):
    def __init__(
        self,
        cfg: dict,
        input_type: str | List[str],
        visual_feature_cfg: dict | None = None,
        cfg_condition: dict | None = None,
    ) -> None:
        super().__init__()
        # cfg sets
        self.encoder_cfg = cfg["encoder"]
        self.mlp_decoder_ctx_cfg = cfg["mlp_decoder_context"]
        self.decoder_cfg = cfg["decoder"]
        self.observation_len = cfg["observation_len"]

        self.noise_dim = self.mlp_decoder_ctx_cfg["noise_dim"]

        if self.encoder_cfg["encoder_type"] == "rnn":
            self.hid_dim = self.encoder_cfg["rnn"]["hidden_dim"]
            self.encoder = RNNEncoder(
                cfg=self.encoder_cfg["rnn"],
                input_type=input_type,
                cfg_condition=cfg_condition,
            )
            self.encoder_kl = RNNEncoder(
                cfg=self.encoder_cfg["rnn"],
                input_type=input_type,
                cfg_condition=cfg_condition,
            )
        elif self.encoder_cfg["encoder_type"] == "tf":
            self.hid_dim = self.encoder_cfg["tf"]["d_model"]
            self.encoder = TFEncoder(cfg=self.encoder_cfg["tf"], input_type=input_type)
            self.encoder_kl = TFEncoder(
                cfg=self.encoder_cfg["tf"], input_type=input_type
            )
            self.reducer_kl = nn.Linear(
                cfg["prediction_len"] * self.noise_dim,
                self.observation_len * self.noise_dim,
            )
        else:
            raise NotImplementedError(self.encoder_cfg["encoder_type"])

        if self.encoder_cfg["encoder_type"] == "tf":
            extra_dim1, extra_dim2 = cfg["prediction_len"], cfg["observation_len"]
        else:
            extra_dim1, extra_dim2 = 1, 1

        self.z_mu = nn.Linear(self.hid_dim * extra_dim1, self.noise_dim * extra_dim2)
        self.z_var = nn.Linear(self.hid_dim * extra_dim1, self.noise_dim * extra_dim2)
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
                    else self.hid_dim * self.observation_len
                ),
                prediction_len=cfg["prediction_len"],
            )
        else:
            raise NotImplementedError(self.decoder_cfg["decoder_type"])

    def sample_noise(
        self,
        batch_size: int,
        hidden_state: Tuple[torch.Tensor, torch.Tensor] | torch.Tensor | None,
    ) -> torch.Tensor:
        if hidden_state is None:  # inference
            if self.encoder_cfg["encoder_type"] == "rnn":
                temp_len = 1
            elif self.encoder_cfg["encoder_type"] == "tf":
                temp_len = self.observation_len
            noise_vector = torch.randn(temp_len, self.noise_dim)
            if self.encoder_cfg["encoder_type"] == "rnn":
                noise_vector = noise_vector.repeat(1, batch_size, 1)
            elif self.encoder_cfg["encoder_type"] == "tf":
                noise_vector = noise_vector.repeat(batch_size, 1, 1)
            return noise_vector, None, None

        if isinstance(hidden_state, tuple):
            hidden_state = hidden_state[0]

        new_hidden_state = (
            hidden_state.squeeze() if hidden_state.shape[0] != 1 else hidden_state
        )
        if self.encoder_cfg["encoder_type"] == "tf":
            new_hidden_state = new_hidden_state.flatten(start_dim=1)
        _mean, _log_var = self.z_mu(new_hidden_state), self.z_var(new_hidden_state)
        if self.encoder_cfg["encoder_type"] == "tf":
            _mean = _mean.view(_mean.size(0), self.observation_len, -1)
            _log_var = _log_var.view(_log_var.size(0), self.observation_len, -1)

        # reparam trick
        std = torch.exp(_log_var / 2)
        q = torch.distributions.Normal(_mean, std)
        random_sample = q.rsample()
        if self.encoder_cfg["encoder_type"] == "rnn":
            random_sample = random_sample.unsqueeze(dim=0)
        return random_sample, _mean, _log_var

    def add_noise(
        self, hidden_cell: torch.Tensor, noise_vector: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        hidden_state, cell_state = (
            hidden_cell if isinstance(hidden_cell, tuple) else (hidden_cell, None)
        )
        new_hidden_state = self.mlp_decoder_context(hidden_state)
        new_hidden_state = torch.cat(
            [new_hidden_state, noise_vector.to(new_hidden_state)], dim=-1
        )
        if cell_state is None:
            return new_hidden_state
        return ((new_hidden_state), (cell_state))

    def forward(self, x: dict, **kwargs):
        encoded_x = self.encoder(x, get_features=False, mask=None, **kwargs)
        bs = x["trajectories"].size(0)
        training_mode = kwargs["y"] is not None
        encoded_y = None
        if training_mode:
            y = kwargs["y"]
            encoded_y = self.encoder_kl(y, get_features=False, mask=None, **kwargs)
        random_noise, mean_, log_var = self.sample_noise(bs, encoded_y)
        noisy_hidden_dim = self.add_noise(encoded_x, random_noise)
        predictions = self.decoder(
            hidden_cell_state=noisy_hidden_dim, x=x, scaler=kwargs["scaler"]
        )
        return (
            predictions.view(predictions.size(0), -1, 2),
            mean_,
            log_var,
            random_noise.squeeze(),
        )


def create_variational_model(
    model_name: str,
    network_cfg: dict,
    input_type: str | List[str],
    visual_feature_cfg: dict | None = None,
):
    if model_name == "vae":
        vae_model = VariationalAutoEncoder(
            cfg=network_cfg,
            input_type=input_type,
            visual_feature_cfg=visual_feature_cfg,
            cfg_condition=None,
        )
    elif model_name == "sup_cvae":
        vae_model = VariationalAutoEncoder(
            cfg=network_cfg,
            input_type=input_type,
            visual_feature_cfg=visual_feature_cfg,
            cfg_condition=network_cfg["condition"],
        )
    return vae_model
