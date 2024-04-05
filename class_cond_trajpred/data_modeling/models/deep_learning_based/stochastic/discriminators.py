from typing import Union, List
import torch
import torch.nn as nn
from ..modules import (
    RNNEncoder,
    FFEncoder,
    TFEncoder,
    MLPDecoder,
)


class Discriminator(nn.Module):
    def __init__(
        self,
        cfg: dict,
        input_type: Union[str, List[str]],
        cfg_condition: dict | None = None,
    ) -> None:
        super().__init__()
        # cfg sets
        self.encoder_cfg = cfg["encoder"]
        self.classifier_cfg = cfg["classifier_head"]

        if self.encoder_cfg["encoder_type"] == "rnn":
            self.hid_dim = self.encoder_cfg["rnn"]["hidden_dim"]
            self.encoder = RNNEncoder(
                cfg=self.encoder_cfg["rnn"],
                input_type=input_type,
                cfg_condition=cfg_condition,
            )
        elif self.encoder_cfg["encoder_type"] == "ff":
            self.encoder = FFEncoder(
                cfg=self.encoder_cfg["ff"],
                input_type=input_type,
                observation_len=cfg["prediction_len"],
                cfg_condition=cfg_condition,
            )
            inp_size_dim_out = self.encoder_cfg["ff"]["mlp_dims"][-1]
        elif self.encoder_cfg["encoder_type"] == "tf":
            self.hid_dim = self.encoder_cfg["tf"]["d_model"]
            self.encoder = TFEncoder(cfg=self.encoder_cfg["tf"], input_type=input_type)
        else:
            raise NotImplementedError(self.encoder_cfg["type"])

        if self.encoder_cfg["encoder_type"] in ["rnn", "tf"]:
            inp_size_dim_out = self.classifier_cfg["mlp_dims"][-1]
            self.classifier_head = MLPDecoder(
                cfg=self.classifier_cfg,
                visual_feature_cfg=None,
                cfg_condition=cfg_condition,
                encoder_hidden_dimension=(
                    self.hid_dim
                    if self.encoder_cfg["encoder_type"] == "rnn"
                    else self.hid_dim * cfg["prediction_len"]
                ),
                prediction_len=None,
            )
        self.fc_out = nn.Linear(inp_size_dim_out, 1)

    def forward(self, traj: torch.Tensor, get_features=False, **kwargs):
        encoded_features = self.encoder(traj, get_features=False, mask=None, **kwargs)
        if self.encoder_cfg["encoder_type"] in ["rnn", "tf"]:
            encoded_features = self.classifier_head(encoded_features, **kwargs)
        if get_features:
            return encoded_features
        return self.fc_out(encoded_features)


def create_discriminator(
    model_name: str, network_cfg: dict, input_type: str | List[str], **kwargs
):
    """returns discriminator if available"""
    if model_name in ["gan", "ft_gan"]:
        discriminator = Discriminator(
            cfg=network_cfg, input_type=input_type, cfg_condition=None
        )
    elif model_name == "sup_cgan":
        discriminator = Discriminator(
            cfg=network_cfg,
            input_type=input_type,
            cfg_condition=network_cfg["condition"],
        )
    else:
        raise NotImplementedError(model_name)
    return discriminator
