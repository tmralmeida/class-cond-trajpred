import os
from copy import deepcopy
import random
from functools import partial
import torch
import torch.optim as optim
import pytorch_lightning as pl
from torch.nn import MSELoss

from class_cond_trajpred.data_modeling.datasets import IOScaler
from class_cond_trajpred.data_modeling.datasets.utils import (
    get_forecasting_input_data,
)
from class_cond_trajpred.evaluation.common_metrics import (
    AverageDisplacementError,
    FinalDisplacementError,
)
from class_cond_trajpred.evaluation.generative_models_metrics import (
    NegativeCondLogLikelihood,
    compute_cll,
)
from class_cond_trajpred.io import dump_json_file
from ..losses import (
    variety_loss,
    feature_matching_loss,
    bce_loss,
    kl_divergence_loss,
    SoftDTW,
)
from .generators import create_generator
from .variational import create_variational_model
from .discriminators import create_discriminator


class LightGANForecaster(pl.LightningModule):
    """GAN-based lightning modules"""

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
        if "clustering_cfg" in kwargs:
            self.clustering_cfg = kwargs["clustering_cfg"]
            self.x_cluster = kwargs["x_cluster"]
            self.data_labels = kwargs["data_labels"]
            saved_hyperparams.update(
                dict(
                    clustering_cfg=self.clustering_cfg,
                    x_cluster=self.x_cluster,
                    data_labels=self.data_labels,
                )
            )
        self.save_hyperparameters(saved_hyperparams)
        gen_cfg, disc_cfg = network_cfg["generator"], network_cfg["discriminator"]
        self.model_name = model_name

        self.generator = create_generator(
            model_name=model_name,
            network_cfg=gen_cfg,
            input_type=data_cfg["inputs"],
            visual_feature_cfg=visual_feature_cfg,
        )
        self.discriminator = create_discriminator(
            model_name=model_name,
            network_cfg=disc_cfg,
            input_type=data_cfg["output"],
        )
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
        self.automatic_optimization = False

        # important hyperparameters
        self.out_type = data_cfg["output"]
        self.gen_clip_threshold = hyperparameters_cfg["g_clip_thresh"]
        self.disc_clip_threshold = hyperparameters_cfg["d_clip_thresh"]
        self.optim_freq_disc = hyperparameters_cfg["optim_freq_disc"]
        self.optim_freq_gen = hyperparameters_cfg["optim_freq_gen"]

        # generator loss function details
        gen_loss = hyperparameters_cfg["generator_loss"]
        tracklet_gen_loss = gen_loss["tracklet_generation"]
        self.track_loss_weight = tracklet_gen_loss["weight"]
        track_loss_name = tracklet_gen_loss["name"]
        if track_loss_name in ["k_variety_mse", "k_variety_soft_dtw"]:
            self.get_tracklet_gen_loss = partial(
                variety_loss,
                n_samples=tracklet_gen_loss["k_train"],
                scaler=self.scaler,
                get_unscaled_outputs=self.get_unscaled_outputs,
                loss=(
                    MSELoss(reduction="none")
                    if track_loss_name == "k_variety_mse"
                    else SoftDTW(use_cuda=visual_feature_cfg is not None, gamma=0.1)
                ),
                loss_weight=self.track_loss_weight,
            )
            self.get_inference_samples = partial(
                variety_loss,
                n_samples=hyperparameters_cfg["n_samples_inference"],
                scaler=self.scaler,
                get_unscaled_outputs=self.get_unscaled_outputs,
                loss=(
                    MSELoss(reduction="none")
                    if track_loss_name == "k_variety_mse"
                    else SoftDTW(use_cuda=visual_feature_cfg is not None, gamma=0.1)
                ),
                loss_weight=self.track_loss_weight,
            )
        else:
            raise NotImplementedError(track_loss_name)
        disc_gen_loss = gen_loss[
            "disc_loss"
        ]  # disc's loss used in the generator optimization
        if disc_gen_loss["name"] == "adversarial":
            self.get_disc_gen_loss = bce_loss
        elif disc_gen_loss["name"] == "feature_matching":
            self.get_disc_gen_loss = feature_matching_loss
        else:
            raise NotImplementedError(disc_gen_loss["name"])
        self.adv_weight = disc_gen_loss["weight"]
        self.topk_prediction = hyperparameters_cfg["n_samples_inference"]
        self.compute_cll = partial(
            compute_cll,
            t_cll=hyperparameters_cfg["t_cll"],
            scaler=self.scaler,
        )

        self.metrics_per_class = (
            {}
            if data_cfg["dataset"] in ["thor", "synthetic", "thor_magni", "sdd"]
            else None
        )
        if self.metrics_per_class is not None:
            self.sup_labels_mapping = data_cfg["supervised_labels"]
            self.n_sup_labels = max(self.sup_labels_mapping.values()) + 1

    def configure_gradient_clipping(
        self, optimizer, optimizer_idx, gradient_clip_val, gradient_clip_algorithm
    ):
        self.clip_gradients(
            optimizer,
            gradient_clip_val=(
                self.gen_clip_threshold
                if optimizer_idx == 0
                else self.disc_clip_threshold
            ),
            gradient_clip_algorithm=gradient_clip_algorithm,
        )

    def configure_optimizers(self):
        g_opt = optim.Adam(
            self.generator.parameters(),
            lr=float(self.hyperparameters_cfg["g_lr"]),
            weight_decay=1e-4,
        )
        d_opt = optim.Adam(
            self.discriminator.parameters(),
            lr=float(self.hyperparameters_cfg["d_lr"]),
            weight_decay=1e-4,
        )
        g_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            g_opt, patience=self.hyperparameters_cfg["scheduler_patience"], min_lr=1e-6
        )
        d_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            d_opt, patience=self.hyperparameters_cfg["scheduler_patience"], min_lr=1e-6
        )
        return [g_opt, d_opt], [g_lr_scheduler, d_lr_scheduler]

    def get_unscaled_outputs(
        self, y_hat: torch.Tensor, obs_tracklet_data: dict
    ) -> torch.Tensor:
        if self.output_type == "trajectories":
            y_hat_unscaled = self.scaler.inv_scale_outputs(y_hat, "trajectories")
        elif self.output_type == "speeds":
            y_hat_unscaled = self.scaler.inv_transform_speeds(y_hat, obs_tracklet_data)
        elif self.output_type == "displacements":
            y_hat_unscaled = self.scaler.inv_transform_displacements(
                y_hat,
                obs_tracklet_data["trajectories"],
            )
        return y_hat_unscaled

    def train_discriminator(
        self,
        optimizer,
        real_tracklets: torch.Tensor,
        generated_tracklets: torch.Tensor,
        batch_size: int,
        batch_idx,
        **kwargs,
    ):
        # soft labels
        real_label = torch.ones((batch_size, 1)) * random.uniform(0.7, 1.2)
        fake_label = torch.ones((batch_size, 1)) * random.uniform(0.0, 0.3)

        fake_loss = bce_loss(
            self.discriminator,
            tracklets=generated_tracklets,
            gt_labels=fake_label,
            labels=kwargs["labels"],
        )
        real_loss = bce_loss(
            self.discriminator,
            tracklets=real_tracklets,
            gt_labels=real_label,
            labels=kwargs["labels"],
        )
        disc_loss = real_loss + fake_loss
        optimizer.zero_grad()
        self.manual_backward(disc_loss)
        if (batch_idx + 1) % self.optim_freq_disc == 0:
            optimizer.step()
        return disc_loss

    def train_generator(
        self,
        optimizer,
        generated_tracklets: torch.Tensor,
        real_tracklets: torch.Tensor,
        k_loss,
        batch_size: int,
        batch_idx: int,
        **kwargs,
    ):
        gen_loss = k_loss.clone() if self.track_loss_weight > 0.0 else torch.zeros(1)
        real_label = torch.ones((batch_size, 1)) * random.uniform(0.7, 1.2)
        disc_loss = self.get_disc_gen_loss(
            discriminator=self.discriminator,
            gt_tracklets=real_tracklets,
            tracklets=generated_tracklets,
            gt_labels=real_label,
            labels=kwargs["labels"],
        )
        gen_loss += self.adv_weight * disc_loss

        optimizer.zero_grad()
        self.manual_backward(gen_loss)
        if (batch_idx + 1) % self.optim_freq_gen == 0:
            optimizer.step()
        return k_loss, gen_loss

    def training_step(self, train_batch: dict, batch_idx: int):
        g_opt, d_opt = self.optimizers()

        obs_tracklet_data, y_gt = self.get_forecasting_data(train_batch)
        batch_size = obs_tracklet_data["trajectories"].size(0)

        # sample from generator
        _, best_generated_tracklets, k_loss, _ = self.get_tracklet_gen_loss(
            obs_tracklet_data=obs_tracklet_data,
            y_gt=y_gt,
            model=self.generator,
        )

        # optimize discriminator
        scaled_gt_batch = self.scaler.scale_inputs(y_gt)[self.out_type]
        disc_loss = self.train_discriminator(
            d_opt,
            scaled_gt_batch,
            best_generated_tracklets.detach(),
            batch_size,
            batch_idx,
            labels=train_batch["data_label"] if self.model_name == "sup_cgan" else None,
        )
        # optimize generator
        k_loss, gen_loss = self.train_generator(
            g_opt,
            best_generated_tracklets,
            scaled_gt_batch,
            k_loss,
            batch_size,
            batch_idx,
            labels=train_batch["data_label"] if self.model_name == "sup_cgan" else None,
        )

        self.log_dict(
            {"mse_loss": k_loss, "g_loss": gen_loss, "d_loss": disc_loss},
            prog_bar=True,
            on_epoch=True,
        )

    def validation_step(self, val_batch: dict, batch_idx: int):
        obs_tracklet_data, y_gt = self.get_forecasting_data(val_batch)
        out_generator = self.get_inference_samples(
            obs_tracklet_data=obs_tracklet_data,
            y_gt=y_gt,
            model=self.generator,
        )
        top1_generated_tracklet, best_generated_tracklets, k_loss, _ = out_generator

        top1_y_hat_unscaled = self.get_unscaled_outputs(
            top1_generated_tracklet.clone(), obs_tracklet_data
        )
        y_hat_unscaled = dict(top1_y_hat=top1_y_hat_unscaled)
        if self.topk_prediction > 1:
            best_y_hat_unscaled = self.get_unscaled_outputs(
                best_generated_tracklets.clone(), obs_tracklet_data
            )
            y_hat_unscaled.update(best_topk_y_hat=best_y_hat_unscaled)

        cll_out = self.compute_cll(
            generator=self.generator,
            obs_tracklet_data=obs_tracklet_data,
            y_gt=y_gt["trajectories"],
            get_unscaled_outputs=self.get_unscaled_outputs,
        )
        self.update_metrics(y_hat_unscaled, y_gt["trajectories"], cll_out)
        self.log("val_loss", k_loss, on_epoch=True, prog_bar=True)

    def test_step(self, test_batch: dict, batch_idx: int):
        obs_tracklet_data, y_gt = self.get_forecasting_data(test_batch)
        out_generator = self.get_inference_samples(
            obs_tracklet_data=obs_tracklet_data,
            y_gt=y_gt,
            model=self.generator,
        )
        top1_generated_tracklet, best_generated_tracklets, _, _ = out_generator

        top1_y_hat_unscaled = self.get_unscaled_outputs(
            top1_generated_tracklet.clone(), obs_tracklet_data
        )
        y_hat_unscaled = dict(top1_y_hat=top1_y_hat_unscaled)
        if self.topk_prediction > 1:
            best_y_hat_unscaled = self.get_unscaled_outputs(
                best_generated_tracklets.clone(), obs_tracklet_data
            )
            y_hat_unscaled.update(best_topk_y_hat=best_y_hat_unscaled)
        cll_out = self.compute_cll(
            generator=self.generator,
            obs_tracklet_data=obs_tracklet_data,
            y_gt=y_gt["trajectories"],
            get_unscaled_outputs=self.get_unscaled_outputs,
        )
        self.update_metrics(y_hat_unscaled, y_gt["trajectories"], cll_out)
        if self.metrics_per_class is not None:
            self.update_metrics_per_class(
                y_hat_unscaled, y_gt["trajectories"], test_batch["data_label"]
            )

    def on_validation_start(self) -> None:
        self.eval_metrics = {
            "CLL": NegativeCondLogLikelihood().to(self.device),
            "Top-1 ADE": AverageDisplacementError().to(self.device),
            "Top-1 FDE": FinalDisplacementError().to(self.device),
        }
        if self.topk_prediction > 1:
            self.eval_metrics.update(
                {
                    f"Top-{self.topk_prediction} ADE": AverageDisplacementError().to(
                        self.device
                    ),
                    f"Top-{self.topk_prediction} FDE": FinalDisplacementError().to(
                        self.device
                    ),
                }
            )

    def on_validation_end(self) -> None:
        save_path = os.path.join(self.logger.log_dir, "val_metrics.json")
        val_metrics = self.compute_metrics()
        dump_json_file(val_metrics, save_path)
        self.reset_metrics()
        g_scheduler, d_scheduler = self.lr_schedulers()
        g_scheduler.step(self.trainer.callback_metrics["val_loss"])
        d_scheduler.step(self.trainer.callback_metrics["val_loss"])

    def on_test_start(self) -> None:
        self.eval_metrics = {
            "CLL": NegativeCondLogLikelihood().to(self.device),
            "Top-1 ADE": AverageDisplacementError().to(self.device),
            "Top-1 FDE": FinalDisplacementError().to(self.device),
        }
        if self.topk_prediction > 1:
            self.eval_metrics.update(
                {
                    f"Top-{self.topk_prediction} ADE": AverageDisplacementError().to(
                        self.device
                    ),
                    f"Top-{self.topk_prediction} FDE": FinalDisplacementError().to(
                        self.device
                    ),
                }
            )
            if self.metrics_per_class is not None:
                for i in range(self.n_sup_labels):
                    self.metrics_per_class[f"Top-{self.topk_prediction} ADE_c{i}"] = (
                        AverageDisplacementError().to(self.device)
                    )
                    self.metrics_per_class[f"Top-{self.topk_prediction} FDE_c{i}"] = (
                        FinalDisplacementError().to(self.device)
                    )

        if self.metrics_per_class is not None:
            for i in range(self.n_sup_labels):
                self.metrics_per_class[f"Top-1 ADE_c{i}"] = (
                    AverageDisplacementError().to(self.device)
                )
                self.metrics_per_class[f"Top-1 FDE_c{i}"] = FinalDisplacementError().to(
                    self.device
                )

    def on_test_end(self) -> None:
        save_path = os.path.join(self.logger.log_dir, "test_metrics.json")
        test_metrics = self.compute_metrics()
        if (
            hasattr(self, "compute_metrics_per_cluster")
            and self.model_name == "self_cgan"
        ):
            test_metrics_cluster = self.compute_metrics_per_cluster()
            test_metrics.update(test_metrics_cluster)
        test_metrics.update(labels_mapping=self.sup_labels_mapping)
        dump_json_file(test_metrics, save_path)
        self.reset_metrics()

    def predict_step(self, predict_batch: dict, batch_idx: int) -> torch.Tensor:
        obs_tracklet_data, _ = self.get_forecasting_data(predict_batch)
        model_input = self.scaler.scale_inputs(obs_tracklet_data)
        outputs = dict()
        outputs["gt"] = predict_batch["trajectories"].detach()
        outputs["y_hat"] = []
        for _ in range(self.topk_prediction):
            y_hat = self.generator(model_input, scaler=self.scaler).detach()
            outputs["y_hat"].append(
                self.get_unscaled_outputs(y_hat.clone(), obs_tracklet_data).detach()
            )

        return outputs

    def update_metrics(self, y_hat: dict, y_gt: torch.Tensor, cll_out: dict):
        top1_y_hat = y_hat["top1_y_hat"]
        topk_y_hat = y_hat["best_topk_y_hat"] if self.topk_prediction > 1 else None
        for metrics_name, metric in self.eval_metrics.items():
            if metrics_name == "CLL":
                metric.update(preds=cll_out["cll_y_hat"], target=cll_out["cll_gt"])
                continue
            metric.update(
                preds=top1_y_hat if metrics_name.startswith("Top-1") else topk_y_hat,
                target=y_gt,
            )

    def update_metrics_per_class(
        self, y_hat: dict, y_gt: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        top1_y_hat = y_hat["top1_y_hat"]
        topk_y_hat = y_hat["best_topk_y_hat"] if self.topk_prediction > 1 else None
        for i, label in enumerate(labels):
            cl_idx = int(label[0].item() if label.size(0) != 1 else label.item())
            self.metrics_per_class[f"Top-1 ADE_c{cl_idx}"].update(
                preds=top1_y_hat[i].unsqueeze(dim=0), target=y_gt[i].unsqueeze(dim=0)
            )
            self.metrics_per_class[f"Top-1 FDE_c{cl_idx}"].update(
                preds=top1_y_hat[i].unsqueeze(dim=0), target=y_gt[i].unsqueeze(dim=0)
            )
            self.metrics_per_class[f"Top-{self.topk_prediction} ADE_c{cl_idx}"].update(
                preds=topk_y_hat[i].unsqueeze(dim=0), target=y_gt[i].unsqueeze(dim=0)
            )
            self.metrics_per_class[f"Top-{self.topk_prediction} FDE_c{cl_idx}"].update(
                preds=topk_y_hat[i].unsqueeze(dim=0), target=y_gt[i].unsqueeze(dim=0)
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


class LightVAEForecaster(pl.LightningModule):
    """VAE-based lightning modules"""

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
        if "clustering_cfg" in kwargs:
            self.clustering_cfg = kwargs["clustering_cfg"]
            self.x_cluster = kwargs["x_cluster"]
            self.data_labels = kwargs["data_labels"]
            saved_hyperparams.update(
                dict(
                    clustering_cfg=self.clustering_cfg,
                    x_cluster=self.x_cluster,
                    data_labels=self.data_labels,
                )
            )
        self.save_hyperparameters(saved_hyperparams)
        self.model_name = model_name
        if self.model_name == "ft_vae":
            ftvae_cfg = deepcopy(network_cfg)
            ftvae_cfg.update(dict(out_type=data_cfg["output"]))
            network_cfg = ftvae_cfg
        self.vae_model = create_variational_model(
            model_name=model_name,
            network_cfg=network_cfg,
            input_type=data_cfg["inputs"],
            visual_feature_cfg=visual_feature_cfg,
        )
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
        self.out_type = data_cfg["output"]
        loss_cfg = hyperparameters_cfg["loss"]
        reconstruction_loss = loss_cfg["reconstruction_loss"]
        regularization_loss = loss_cfg["regularization_loss"]
        loss_name = reconstruction_loss["name"]
        if reconstruction_loss["name"] in ["k_variety_mse", "k_variety_soft_dtw"]:
            self.get_tracklet_gen_loss = partial(
                variety_loss,
                n_samples=reconstruction_loss["k_train"],
                scaler=self.scaler,
                get_unscaled_outputs=self.get_unscaled_outputs,
                loss=(
                    MSELoss(reduction="none")
                    if loss_name == "k_variety_mse"
                    else SoftDTW(use_cuda=visual_feature_cfg is not None, gamma=0.1)
                ),
                loss_weight=reconstruction_loss["weight"],
            )
            self.get_inference_samples = partial(
                variety_loss,
                n_samples=hyperparameters_cfg["n_samples_inference"],
                scaler=self.scaler,
                get_unscaled_outputs=self.get_unscaled_outputs,
                loss=(
                    MSELoss(reduction="none")
                    if loss_name == "k_variety_mse"
                    else SoftDTW(use_cuda=visual_feature_cfg is not None, gamma=0.1)
                ),
                loss_weight=reconstruction_loss["weight"],
            )
        else:
            raise NotImplementedError(loss_name)
        if regularization_loss["name"] == "kl_divergence":
            self.get_regularization_loss = partial(
                kl_divergence_loss, weight=regularization_loss["weight"]
            )
        else:
            raise NotImplementedError(regularization_loss["name"])
        self.topk_prediction = hyperparameters_cfg["n_samples_inference"]
        self.compute_cll = partial(
            compute_cll,
            t_cll=hyperparameters_cfg["t_cll"],
            scaler=self.scaler,
            get_unscaled_outputs=self.get_unscaled_outputs,
        )
        self.metrics_per_class = (
            {}
            if data_cfg["dataset"] in ["thor", "synthetic", "thor_magni", "sdd"]
            else None
        )
        if self.metrics_per_class is not None:
            self.sup_labels_mapping = data_cfg["supervised_labels"]
            self.n_sup_labels = max(self.sup_labels_mapping.values()) + 1

    def get_unscaled_outputs(
        self, y_hat: torch.Tensor, obs_tracklet_data: dict
    ) -> torch.Tensor:
        if self.output_type == "trajectories":
            y_hat_unscaled = self.scaler.inv_scale_outputs(y_hat, "trajectories")
        elif self.output_type == "speeds":
            y_hat_unscaled = self.scaler.inv_transform_speeds(y_hat, obs_tracklet_data)
        elif self.output_type == "displacements":
            y_hat_unscaled = self.scaler.inv_transform_displacements(
                y_hat,
                obs_tracklet_data["trajectories"],
            )
        return y_hat_unscaled

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
            dict(scheduler=lr_scheduler, interval="epoch", monitor="rec_loss")
        ]

    def training_step(self, train_batch: dict, batch_idx: int) -> torch.Tensor:
        obs_tracklet_data, y_gt = self.get_forecasting_data(train_batch)

        # reconstruction loss
        _, _, k_loss, model_stats = self.get_tracklet_gen_loss(
            obs_tracklet_data=obs_tracklet_data,
            y_gt=y_gt,
            model=self.vae_model,
        )

        # regularization loss
        reg_loss = self.get_regularization_loss(
            mu=model_stats["mu"], log_var=model_stats["log_var"]
        )

        loss = k_loss + reg_loss

        self.log_dict(
            {"rec_loss": k_loss, "reg_loss": reg_loss},
            prog_bar=True,
            on_epoch=True,
        )
        return loss

    def validation_step(self, val_batch: dict, batch_idx: int):
        obs_tracklet_data, y_gt = self.get_forecasting_data(val_batch)
        out_vae = self.get_inference_samples(
            obs_tracklet_data=obs_tracklet_data,
            y_gt=y_gt,
            model=self.vae_model,
        )
        top1_gen_tracklet, best_gen_tracklets, k_loss, _ = out_vae

        loss = k_loss

        top1_y_hat_unscaled = self.get_unscaled_outputs(
            top1_gen_tracklet.clone(), obs_tracklet_data
        )
        y_hat_unscaled = dict(top1_y_hat=top1_y_hat_unscaled)
        if self.topk_prediction > 1:
            best_y_hat_unscaled = self.get_unscaled_outputs(
                best_gen_tracklets.clone(), obs_tracklet_data
            )
            y_hat_unscaled.update(best_topk_y_hat=best_y_hat_unscaled)
        cll_out = self.compute_cll(
            generator=self.vae_model,
            obs_tracklet_data=obs_tracklet_data,
            y_gt=y_gt["trajectories"],
        )
        self.update_metrics(y_hat_unscaled, y_gt["trajectories"], cll_out)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def test_step(self, test_batch: dict, batch_idx: int):
        obs_tracklet_data, y_gt = self.get_forecasting_data(test_batch)
        out_vae = self.get_inference_samples(
            obs_tracklet_data=obs_tracklet_data,
            y_gt=y_gt,
            model=self.vae_model,
        )
        top1_generated_tracklet, best_generated_tracklets, _, _ = out_vae

        top1_y_hat_unscaled = self.get_unscaled_outputs(
            top1_generated_tracklet.clone(), obs_tracklet_data
        )
        y_hat_unscaled = dict(top1_y_hat=top1_y_hat_unscaled)
        if self.topk_prediction > 1:
            best_y_hat_unscaled = self.get_unscaled_outputs(
                best_generated_tracklets.clone(), obs_tracklet_data
            )
            y_hat_unscaled.update(best_topk_y_hat=best_y_hat_unscaled)
        cll_out = self.compute_cll(
            generator=self.vae_model,
            obs_tracklet_data=obs_tracklet_data,
            y_gt=y_gt["trajectories"],
        )
        self.update_metrics(y_hat_unscaled, y_gt["trajectories"], cll_out)
        if self.metrics_per_class is not None:
            self.update_metrics_per_class(
                y_hat_unscaled, y_gt["trajectories"], test_batch["data_label"]
            )

    def on_validation_start(self) -> None:
        self.eval_metrics = {
            "CLL": NegativeCondLogLikelihood().to(self.device),
            "Top-1 ADE": AverageDisplacementError().to(self.device),
            "Top-1 FDE": FinalDisplacementError().to(self.device),
        }
        if self.topk_prediction > 1:
            self.eval_metrics.update(
                {
                    f"Top-{self.topk_prediction} ADE": AverageDisplacementError().to(
                        self.device
                    ),
                    f"Top-{self.topk_prediction} FDE": FinalDisplacementError().to(
                        self.device
                    ),
                }
            )

    def on_validation_end(self) -> None:
        save_path = os.path.join(self.logger.log_dir, "val_metrics.json")
        val_metrics = self.compute_metrics()
        dump_json_file(val_metrics, save_path)
        self.reset_metrics()

    def on_test_start(self) -> None:
        self.eval_metrics = {
            "CLL": NegativeCondLogLikelihood().to(self.device),
            "Top-1 ADE": AverageDisplacementError().to(self.device),
            "Top-1 FDE": FinalDisplacementError().to(self.device),
        }
        if self.topk_prediction > 1:
            self.eval_metrics.update(
                {
                    f"Top-{self.topk_prediction} ADE": AverageDisplacementError().to(
                        self.device
                    ),
                    f"Top-{self.topk_prediction} FDE": FinalDisplacementError().to(
                        self.device
                    ),
                }
            )

            if self.metrics_per_class is not None:
                for i in range(self.n_sup_labels):
                    self.metrics_per_class[f"Top-{self.topk_prediction} ADE_c{i}"] = (
                        AverageDisplacementError().to(self.device)
                    )
                    self.metrics_per_class[f"Top-{self.topk_prediction} FDE_c{i}"] = (
                        FinalDisplacementError().to(self.device)
                    )

        if self.metrics_per_class is not None:
            for i in range(self.n_sup_labels):
                self.metrics_per_class[f"Top-1 ADE_c{i}"] = (
                    AverageDisplacementError().to(self.device)
                )
                self.metrics_per_class[f"Top-1 FDE_c{i}"] = FinalDisplacementError().to(
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
        obs_tracklet_data, _ = self.get_forecasting_data(predict_batch)
        model_input = self.scaler.scale_inputs(obs_tracklet_data)
        outputs = dict()
        outputs["gt"] = predict_batch["trajectories"].detach()
        outputs["y_hat"] = []
        for _ in range(self.topk_prediction):
            y_hat = self.vae_model(model_input, scaler=self.scaler, y=None)[0].detach()
            outputs["y_hat"].append(
                self.get_unscaled_outputs(y_hat.clone(), obs_tracklet_data).detach()
            )

        return outputs

    def update_metrics(self, y_hat: dict, y_gt: torch.Tensor, cll_out: dict):
        top1_y_hat = y_hat["top1_y_hat"]
        topk_y_hat = y_hat["best_topk_y_hat"] if self.topk_prediction > 1 else None
        for metrics_name, metric in self.eval_metrics.items():
            if metrics_name == "CLL":
                metric.update(preds=cll_out["cll_y_hat"], target=cll_out["cll_gt"])
                continue
            metric.update(
                preds=top1_y_hat if metrics_name.startswith("Top-1") else topk_y_hat,
                target=y_gt,
            )

    def update_metrics_per_class(
        self, y_hat: dict, y_gt: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        top1_y_hat = y_hat["top1_y_hat"]
        topk_y_hat = y_hat["best_topk_y_hat"] if self.topk_prediction > 1 else None
        for i, label in enumerate(labels):
            cl_idx = int(label[0].item() if label.size(0) != 1 else label.item())
            self.metrics_per_class[f"Top-1 ADE_c{cl_idx}"].update(
                preds=top1_y_hat[i].unsqueeze(dim=0), target=y_gt[i].unsqueeze(dim=0)
            )
            self.metrics_per_class[f"Top-1 FDE_c{cl_idx}"].update(
                preds=top1_y_hat[i].unsqueeze(dim=0), target=y_gt[i].unsqueeze(dim=0)
            )
            self.metrics_per_class[f"Top-{self.topk_prediction} ADE_c{cl_idx}"].update(
                preds=topk_y_hat[i].unsqueeze(dim=0), target=y_gt[i].unsqueeze(dim=0)
            )
            self.metrics_per_class[f"Top-{self.topk_prediction} FDE_c{cl_idx}"].update(
                preds=topk_y_hat[i].unsqueeze(dim=0), target=y_gt[i].unsqueeze(dim=0)
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
