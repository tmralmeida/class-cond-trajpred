from copy import copy
from typing import List, Union, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..modules import (
    Encoder,
    make_mlp,
    CNNFeatureExtractor,
    LatentEmbedding,
    cat_class_emb,
)


class RecurrentNetwork(nn.Module):
    """Recurrent networks based models: GRU LSTM. Similar to RED predictor but with visual feature
    encoder if that's available"""

    def __init__(
        self,
        cfg: dict,
        input_type: Union[str, List[str]],
        visual_feature_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.hid_dim = cfg["hidden_dim"]
        self.emb_dim = cfg["embedding_dim"]
        self.activ = cfg["activation"]
        self.drop = cfg["dropout"]
        self.network_type = cfg["type"]
        self.hidden_state = True if cfg["state"] == "hidden" else False
        self.input_type = input_type if isinstance(input_type, list) else [input_type]
        self.input_dims = sum(
            [1 if inp == "straightness_index" else 2 for inp in self.input_type]
        )

        self.visual_out_size = 0
        if visual_feature_cfg:
            visual_feature_cfg = visual_feature_cfg["cnn_feature_extractor"]
            self.visual_feature_encoder = CNNFeatureExtractor(visual_feature_cfg)
            self.visual_out_size = visual_feature_cfg["linear_block"]["out_size"]

        mlp_dims = copy(cfg["mlp_dims"])
        mlp_dims.insert(0, self.hid_dim + self.visual_out_size)
        mlp_dims.append(cfg["prediction_len"] * 2)

        self.encoder = Encoder(
            inp_dim=self.input_dims,
            emb_dim=self.emb_dim,
            hid_dim=self.hid_dim,
            class_emb_dim=0,
            network_type=self.network_type,
        )
        self.decoder = make_mlp(
            dim_list=mlp_dims, activation=self.activ, dropout=self.drop
        )

    def forward(self, x: dict) -> torch.Tensor:
        inputs_cat = []
        for feature_name, inputs in x.items():
            if feature_name in self.input_type:
                inputs = inputs if inputs.dim() == 3 else inputs.unsqueeze(dim=-1)
                inputs_cat.append(inputs)
        inputs_cat = torch.cat(inputs_cat, dim=-1)
        bs = inputs.size(0)
        hidden_cell_init = (
            (
                (torch.zeros(1, bs, self.hid_dim)).to(inputs_cat),
                (torch.zeros(1, bs, self.hid_dim)).to(inputs_cat),
            )
            if self.network_type == "lstm"
            else torch.zeros(1, bs, self.hid_dim)
        )

        # encoder
        inp_lstm = self.encoder.input_embedding(
            inputs_cat.contiguous().view(-1, self.input_dims)
        )
        inp_lstm = inp_lstm.view(bs, -1, self.emb_dim)
        _, hidden_cell = self.encoder.temp_feat(inp_lstm, hidden_cell_init)
        hn = (
            hidden_cell[0 if self.hidden_state else 1]
            if self.network_type == "lstm"
            else hidden_cell
        )
        hn = hn.view(-1, self.hid_dim)
        if self.visual_out_size > 0:
            out_features = self.visual_feature_encoder(x["img"])
            hn = torch.cat([hn, out_features], dim=-1)

        # decoder
        out = self.decoder(hn)
        return out.view(bs, -1, 2)


class SupCondRecurrentNetwork(RecurrentNetwork):
    def __init__(
        self,
        cfg: dict,
        input_type: str | List[str],
        visual_feature_cfg: dict | None = None,
    ) -> None:
        super().__init__(cfg, input_type, visual_feature_cfg)
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

        mlp_dims = copy(cfg["mlp_dims"])
        mlp_dims.insert(0, self.hid_dim + self.visual_out_size + self.class_emb_dim)
        mlp_dims.append(cfg["prediction_len"] * 2)

        self.decoder = make_mlp(
            dim_list=mlp_dims, activation=self.activ, dropout=self.drop
        )

    def forward(self, x: dict) -> torch.Tensor:
        inputs_cat = []
        for feature_name, inputs in x.items():
            if feature_name in self.input_type:
                inputs = inputs if inputs.dim() == 3 else inputs.unsqueeze(dim=-1)
                inputs_cat.append(inputs)
        inputs_cat = torch.cat(inputs_cat, dim=-1)
        bs = inputs.size(0)
        hidden_cell_init = (
            (
                (torch.zeros(1, bs, self.hid_dim)).to(inputs_cat),
                (torch.zeros(1, bs, self.hid_dim)).to(inputs_cat),
            )
            if self.network_type == "lstm"
            else torch.zeros(1, bs, self.hid_dim)
        )

        # encoder
        inp_lstm = self.encoder.input_embedding(
            inputs_cat.contiguous().view(-1, self.input_dims)
        )
        inp_lstm = inp_lstm.view(bs, -1, self.emb_dim)
        _, hidden_cell = self.encoder.temp_feat(inp_lstm, hidden_cell_init)
        hn = (
            hidden_cell[0 if self.hidden_state else 1]
            if self.network_type == "lstm"
            else hidden_cell
        )
        hn = hn.view(-1, self.hid_dim)
        if self.visual_out_size > 0:
            out_features = self.visual_feature_encoder(x["img"])
            hn = torch.cat([hn, out_features], dim=-1)

        labels = x["data_label"][:, 0].long()
        labels = (
            self.emb_layer(labels)
            if self.cond_type == "embedding"
            else F.one_hot(labels, num_classes=self.n_classes).float()
        )

        hn = cat_class_emb(hn, labels)

        # decoder
        out = self.decoder(hn)
        return out.view(bs, -1, 2)
