from typing import Union, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_

from .tf_modules import PositionalEncoding, TransformerEncoder
from ..modules import LatentEmbedding, cat_class_emb


class TransformerEncMLP(nn.Module):
    def __init__(
        self,
        cfg: dict,
        input_type: Union[str, List[str]],
    ) -> None:
        super().__init__()
        self.input_type = input_type if isinstance(input_type, list) else [input_type]
        self.d_model = cfg["d_model"]
        input_dims = sum(
            [1 if inp == "straightness_index" else 2 for inp in self.input_type]
        )
        self.emb_net = nn.Sequential(
            nn.Linear(input_dims, self.d_model), nn.Dropout(cfg["dropout"])
        )
        self.positional_encoding = PositionalEncoding(d_model=self.d_model)
        self.transformer_encoder = TransformerEncoder(
            num_layers=cfg["num_layers"],
            input_dim=self.d_model,
            dim_feedforward=2 * self.d_model,
            num_heads=cfg["num_heads"],
            dropout=cfg["dropout"],
        )
        intermediate_size = self.d_model * cfg["observation_len"] // 2
        self.output_net = nn.Sequential(
            nn.Linear(self.d_model * cfg["observation_len"], intermediate_size),
            nn.Dropout(cfg["dropout"]),
            nn.ReLU(inplace=True),
            nn.Linear(intermediate_size, 2 * cfg["prediction_len"]),
        )
        self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, x: dict, mask: Optional[torch.Tensor] = None):
        inputs_cat = []
        for feature_name, inputs in x.items():
            if feature_name in self.input_type:
                inputs = inputs if inputs.dim() == 3 else inputs.unsqueeze(dim=-1)
                inputs_cat.append(inputs)
        inputs_cat = torch.cat(inputs_cat, dim=-1)
        bs = inputs_cat.size(0)
        x = self.emb_net(inputs_cat)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x, mask=mask)
        x = x.flatten(start_dim=1)
        x = self.output_net(x)
        return x.view(bs, -1, 2)


class SupCondTransformerEncMLP(TransformerEncMLP):
    def __init__(self, cfg: dict, input_type: str | List[str]) -> None:
        super().__init__(cfg, input_type)
        cfg_condition = cfg["condition"]
        self.n_classes = cfg_condition["n_labels"]
        self.cond_type = cfg_condition["name"]
        if self.cond_type not in ["embedding", "one_hot"]:
            raise NotImplementedError(self.cond_type)
        self.class_emb_dim = (
            cfg_condition["embedding_dim"]
            if self.cond_type == "embedding"
            else self.n_classes
        )
        self.emb_layer = (
            LatentEmbedding(self.n_classes, self.class_emb_dim)
            if self.cond_type == "embedding"
            else None
        )
        intermediate_size = (
            self.d_model * cfg["observation_len"] + self.class_emb_dim
        ) // 2
        self.output_net = nn.Sequential(
            nn.Linear(
                self.d_model * cfg["observation_len"] + self.class_emb_dim,
                intermediate_size,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                intermediate_size,
                2 * cfg["prediction_len"],
            ),
        )

    def forward(self, x: dict, mask: Optional[torch.Tensor] = None):
        labels = x["data_label"][:, 0].long()
        labels = (
            self.emb_layer(labels)
            if self.cond_type == "embedding"
            else F.one_hot(labels, num_classes=self.n_classes).float()
        )
        inputs_cat = []
        for feature_name, inputs in x.items():
            if feature_name in self.input_type:
                inputs = inputs if inputs.dim() == 3 else inputs.unsqueeze(dim=-1)
                inputs_cat.append(inputs)
        inputs_cat = torch.cat(inputs_cat, dim=-1)
        bs = inputs_cat.size(0)
        x = self.emb_net(inputs_cat)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x, mask=mask)
        x = x.flatten(start_dim=1)
        hn = cat_class_emb(x, labels)
        x = self.output_net(hn)
        return x.view(bs, -1, 2)
