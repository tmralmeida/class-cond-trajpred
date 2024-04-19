import os
from functools import partial
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from ....datasets import IOScaler
from ....datasets.utils import get_forecasting_input_data
from .....evaluation.common_metrics import (
    AverageDisplacementError,
    FinalDisplacementError,
)
from .....io import dump_json_file
from .recurrent_models import RecurrentNetwork, SupCondRecurrentNetwork
from .tf_models import TransformerEncMLP, SupCondTransformerEncMLP
from ..losses import SoftDTW


class LightPointForecaster(pl.LightningModule):
    """LSTM, GRU, and Transformers lightning modules"""

    def __init__(self, **kwargs) -> None:
        super().__init__()
        model_name = kwargs["model_name"]
        data_cfg = kwargs["data_cfg"]
        network_cfg = kwargs["network_cfg"]
        hyperparameters_cfg = kwargs["hyperparameters_cfg"]
        visual_feature_cfg = kwargs["visual_feature_cfg"]
        saved_hyperparams = dict(
            model_name=model_name,
            data_cfg=data_cfg,
            network_cfg=network_cfg,
            hyperparameters_cfg=hyperparameters_cfg,
            visual_feature_cfg=visual_feature_cfg,
        )
        self.save_hyperparameters(saved_hyperparams)

        if model_name == "rnn":
            self.model = RecurrentNetwork(
                cfg=network_cfg,
                input_type=data_cfg["inputs"],
                visual_feature_cfg=visual_feature_cfg,
            )
        elif model_name == "sup_crnn":
            self.model = SupCondRecurrentNetwork(
                cfg=network_cfg,
                input_type=data_cfg["inputs"],
                visual_feature_cfg=visual_feature_cfg,
            )
        elif model_name == "tf":
            # TODO: include visual features
            self.model = TransformerEncMLP(
                cfg=network_cfg,
                input_type=data_cfg["inputs"],
            )
        elif model_name == "sup_ctf":
            # TODO: include visual features
            self.model = SupCondTransformerEncMLP(
                cfg=network_cfg, input_type=data_cfg["inputs"]
            )
        else:
            raise NotImplementedError(model_name)
        self.hyperparameters_cfg = hyperparameters_cfg
        scaler_kwargs = {}
        if "fold_index" in kwargs.keys():
            fold_index = kwargs["fold_index"]
            scaler_kwargs.update(fold_index=fold_index)
        self.scaler = IOScaler(data_cfg, **scaler_kwargs)
        self.output_type = data_cfg["output"]
        self.get_forecasting_data = partial(
            get_forecasting_input_data,
            obs_len=data_cfg["observation_len"],
            inputs=data_cfg["inputs"],
        )
        loss_type = self.hyperparameters_cfg["loss_type"]
        if loss_type == "mse":
            self.loss = nn.MSELoss(reduction="none")
        elif loss_type == "soft_dtw":
            self.loss = SoftDTW(use_cuda=visual_feature_cfg is not None, gamma=0.1)
        else:
            raise NotImplementedError(loss_type)
        self.metrics_per_class = (
            {}
            if data_cfg["dataset"] in ["thor", "synthetic", "thor_magni", "sdd"]
            else None
        )
        if self.metrics_per_class is not None:
            self.sup_labels_mapping = data_cfg["supervised_labels"]
            self.n_sup_labels = max(self.sup_labels_mapping.values()) + 1

    def forward(self, x):
        y_hat = self.model(x)
        return y_hat

    def configure_optimizers(self):
        opt = optim.Adam(
            self.parameters(),
            lr=float(self.hyperparameters_cfg["lr"]),
            weight_decay=1e-4,
        )
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt, patience=self.hyperparameters_cfg["scheduler_patience"], min_lr=1e-6
        )
        return [opt], [
            dict(scheduler=lr_scheduler, interval="epoch", monitor="train_loss")
        ]

    def training_step(self, train_batch: dict, batch_idx: int) -> torch.Tensor:
        y_gt, y_hat_unscaled = self.common_step(train_batch)
        loss = self.loss(y_hat_unscaled, y_gt["trajectories"]).mean()
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, val_batch: dict, batch_idx: int) -> torch.Tensor:
        y_gt, y_hat_unscaled = self.common_step(val_batch)
        self.update_metrics(y_hat_unscaled, y_gt["trajectories"])
        val_loss = self.loss(y_hat_unscaled, y_gt["trajectories"]).mean()
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True)

    def test_step(self, test_batch: dict, batch_idx: int) -> torch.Tensor:
        y_gt, y_hat_unscaled = self.common_step(test_batch)
        self.update_metrics(y_hat_unscaled, y_gt["trajectories"])
        if self.metrics_per_class is not None:
            self.update_metrics_per_class(
                y_hat_unscaled, y_gt["trajectories"], test_batch["data_label"]
            )

    def on_validation_start(self) -> None:
        self.eval_metrics = dict(
            ade=AverageDisplacementError().to(self.device),
            fde=FinalDisplacementError().to(self.device),
        )

    def on_validation_end(self) -> None:
        save_path = os.path.join(self.logger.log_dir, "val_metrics.json")
        val_metrics = self.compute_metrics()
        dump_json_file(val_metrics, save_path)
        self.reset_metrics()

    def on_test_start(self) -> None:
        self.eval_metrics = dict(
            ade=AverageDisplacementError().to(self.device),
            fde=FinalDisplacementError().to(self.device),
        )
        if self.metrics_per_class is not None:
            for i in range(self.n_sup_labels):
                self.metrics_per_class[f"ADE_c{i}"] = AverageDisplacementError().to(
                    self.device
                )
                self.metrics_per_class[f"FDE_c{i}"] = FinalDisplacementError().to(
                    self.device
                )

    def on_test_end(self) -> None:
        save_path = os.path.join(self.logger.log_dir, "test_metrics.json")
        test_metrics = self.compute_metrics()
        if self.metrics_per_class is not None:
            test_metrics.update(labels_mapping=self.sup_labels_mapping)
        dump_json_file(test_metrics, save_path)
        self.reset_metrics()

    def predict_step(self, predict_batch: dict, batch_idx: int) -> torch.Tensor:
        _, y_hat_unscaled = self.common_step(predict_batch)
        return dict(
            gt=predict_batch["trajectories"].detach(),
            y_hat=[y_hat_unscaled.detach()],
        )

    def common_step(self, batch: dict):
        obs_tracklet_data, y_gt = self.get_forecasting_data(batch)
        scaled_train_batch = self.scaler.scale_inputs(obs_tracklet_data)
        y_hat = self(scaled_train_batch).clone()
        if self.output_type == "trajectories":
            y_hat_unscaled = self.scaler.inv_scale_outputs(y_hat, "trajectories")
        elif self.output_type == "speeds":
            y_hat_unscaled = self.scaler.inv_transform_speeds(y_hat, obs_tracklet_data)
        elif self.output_type == "displacements":
            y_hat_unscaled = self.scaler.inv_transform_displacements(
                y_hat,
                obs_tracklet_data["trajectories"],
            )
        return y_gt, y_hat_unscaled

    def update_metrics(self, y_hat: torch.Tensor, y_gt: torch.Tensor):
        for _, metric in self.eval_metrics.items():
            metric.update(preds=y_hat, target=y_gt)

    def update_metrics_per_class(
        self, y_hat: torch.Tensor, y_gt: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        for i, label in enumerate(labels):
            cl_idx = int(label[0].item() if label.size(0) != 1 else label.item())
            self.metrics_per_class[f"ADE_c{cl_idx}"].update(
                preds=y_hat[i].unsqueeze(dim=0), target=y_gt[i].unsqueeze(dim=0)
            )
            self.metrics_per_class[f"FDE_c{cl_idx}"].update(
                preds=y_hat[i].unsqueeze(dim=0), target=y_gt[i].unsqueeze(dim=0)
            )

    def compute_metrics(self) -> dict:
        final_metrics = {
            met_name: met.compute().item()
            for met_name, met in self.eval_metrics.items()
        }
        if self.metrics_per_class is not None:
            final_labels_metrics = {
                met_name: met.compute().item()
                for met_name, met in self.metrics_per_class.items()
            }
            final_metrics.update(final_labels_metrics)
        return final_metrics

    def reset_metrics(self) -> None:
        for _, metric in self.eval_metrics.items():
            metric.reset()
        if self.metrics_per_class is not None:
            for _, metric in self.metrics_per_class.items():
                metric.reset()
