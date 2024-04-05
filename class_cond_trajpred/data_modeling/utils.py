import os
import pandas as pd
from typing import Optional
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)

from class_cond_trajpred.io import load_yaml_file
from .models.deep_learning_based import LightPointForecaster
from .models.deep_learning_based import (
    LightGANForecaster,
    LightVAEForecaster,
)
from .datasets import load_dataset
from .datasets.torch_loaders import ThorDataset, SDDDataset


def get_trainer_objects(
    cfg: dict, accelerator: str = "cpu", subdir: Optional[str] = None
):
    model_name = cfg["model"]
    data_cfg = cfg["data"]
    network_cfg = cfg["network"]
    hyperparameters_cfg = cfg["hyperparameters"]
    save_cfg = cfg["save"]
    dataset_name = data_cfg["dataset"]
    dataset_target = data_cfg["dataset_target"]

    save_path = (
        os.path.join(save_cfg["path"], dataset_name, dataset_target)
        if dataset_target
        else os.path.join(save_cfg["path"], dataset_name)
    )

    visual_feature_extractor_cfg = cfg["visual_feature_extractor"]
    vis_features_cfg = None
    if visual_feature_extractor_cfg["use"]:
        vis_features_cfg = load_yaml_file(visual_feature_extractor_cfg["inherit_from"])
        visuals_path = vis_features_cfg["data_dir"]
        visual_window_size = vis_features_cfg["window_size"]
        data_cfg.update(dict(visuals_path=visuals_path, window_size=visual_window_size))

    train_ds = load_dataset("train", data_cfg)
    val_ds = load_dataset("val", data_cfg)
    test_ds = load_dataset("test", data_cfg)
    if dataset_name in ["thor", "thor_magni", "sdd"]:
        val_ds.mapping_roles = train_ds.mapping_roles
        test_ds.mapping_roles = train_ds.mapping_roles
        data_cfg.update(dict(supervised_labels=dict(train_ds.mapping_roles)))
    elif dataset_name == "synthetic":
        data_cfg.update(dict(supervised_labels=dict(train_ds.mapping_modes)))

    train_dl = DataLoader(train_ds, hyperparameters_cfg["bs"], shuffle=True)
    val_dl = DataLoader(val_ds, hyperparameters_cfg["bs"], shuffle=False)
    test_dl = DataLoader(test_ds, hyperparameters_cfg["bs"], shuffle=False)

    subdir = os.path.join(model_name, subdir) if subdir else model_name
    logger = TensorBoardLogger(save_path, name=subdir, default_hp_metric=False)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", save_top_k=1, filename="{epoch}-{val_loss:.2f}"
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.01,
        patience=hyperparameters_cfg["patience"],
        verbose=False,
        mode="min",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [early_stop_callback, checkpoint_callback, lr_monitor]
    model_name = cfg["model"]
    if model_name in ["rnn", "sup_crnn", "tf", "sup_ctf"]:
        lightning_module = LightPointForecaster(
            model_name=model_name,
            data_cfg=data_cfg,
            network_cfg=network_cfg,
            hyperparameters_cfg=hyperparameters_cfg,
            visual_feature_cfg=vis_features_cfg,
        )
    elif model_name in ["gan", "sup_cgan"]:
        lightning_module = LightGANForecaster(
            model_name=model_name,
            data_cfg=data_cfg,
            network_cfg=network_cfg,
            hyperparameters_cfg=hyperparameters_cfg,
            visual_feature_cfg=vis_features_cfg,
        )
    elif model_name in ["vae", "sup_cvae"]:
        lightning_module = LightVAEForecaster(
            model_name=model_name,
            data_cfg=data_cfg,
            network_cfg=network_cfg,
            hyperparameters_cfg=hyperparameters_cfg,
            visual_feature_cfg=vis_features_cfg,
        )
    else:
        raise NotImplementedError(model_name)
    trainer = pl.Trainer(
        default_root_dir=save_cfg["path"],
        logger=logger,
        accelerator=accelerator,
        callbacks=callbacks,
        max_epochs=hyperparameters_cfg["max_epochs"],
        check_val_every_n_epoch=hyperparameters_cfg["val_freq"],
    )
    test_dl = test_dl if data_cfg["test"] else None
    return trainer, lightning_module, train_dl, val_dl, test_dl


def run_trainer(trainer_options):
    trainer, lightning_module, train_dl, val_dl, test_dl = trainer_options
    trainer.fit(lightning_module, train_dl, val_dl)
    if test_dl:
        trainer.test(ckpt_path="best", dataloaders=test_dl)
    return trainer.logger.log_dir


def get_trainer_objects_kfold_cv_from_yufei(
    data_path: str,
    fold_index: int,
    cfg: dict,
    accelerator: str = "cpu",
    subdir: Optional[str] = None,
):
    model_name = cfg["model"]
    data_cfg = cfg["data"]
    network_cfg = cfg["network"]
    hyperparameters_cfg = cfg["hyperparameters"]
    save_cfg = cfg["save"]
    dataset_name = data_cfg["dataset"]
    dataset_target = data_cfg["dataset_target"]
    save_path = (
        os.path.join(save_cfg["path"], dataset_name, dataset_target)
        if dataset_target
        else os.path.join(save_cfg["path"], dataset_name)
    )

    visual_feature_extractor_cfg = cfg["visual_feature_extractor"]
    vis_features_cfg = None
    if visual_feature_extractor_cfg["use"]:
        vis_features_cfg = load_yaml_file(visual_feature_extractor_cfg["inherit_from"])
        visuals_path = vis_features_cfg["data_dir"]
        visual_window_size = vis_features_cfg["window_size"]
        data_cfg.update(dict(visuals_path=visuals_path, window_size=visual_window_size))

    rel_path = os.path.join(data_path, dataset_target) if dataset_target else data_path
    train_trajectories = pd.read_pickle(
        os.path.join(rel_path, f"train_b{fold_index}.pkl")
    )
    test_trajectories = pd.read_pickle(
        os.path.join(rel_path, f"test_b{fold_index}.pkl")
    )

    if dataset_name == "thor_magni":
        train_ds = ThorDataset(
            train_trajectories, collapse_visitors=data_cfg["collapse_visitors"]
        )
        val_ds = ThorDataset(
            test_trajectories, collapse_visitors=data_cfg["collapse_visitors"]
        )
    elif dataset_name == "sdd":
        train_ds = SDDDataset(train_trajectories)
        val_ds = SDDDataset(
            test_trajectories,
        )
    val_ds.mapping_roles = train_ds.mapping_roles
    data_cfg.update(dict(supervised_labels=train_ds.mapping_roles))

    train_dl = DataLoader(train_ds, hyperparameters_cfg["bs"], shuffle=True)
    val_dl = DataLoader(val_ds, hyperparameters_cfg["bs"], shuffle=False)

    subdir = os.path.join(model_name, subdir) if subdir else model_name
    logger = TensorBoardLogger(save_path, name=subdir, default_hp_metric=False)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", save_top_k=1, filename="{epoch}-{val_loss:.2f}"
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.01,
        patience=hyperparameters_cfg["patience"],
        verbose=False,
        mode="min",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [early_stop_callback, checkpoint_callback, lr_monitor]
    model_name = cfg["model"]
    if model_name in ["rnn", "sup_crnn", "tf", "sup_ctf"]:
        lightning_module = LightPointForecaster(
            model_name=model_name,
            data_cfg=data_cfg,
            network_cfg=network_cfg,
            hyperparameters_cfg=hyperparameters_cfg,
            visual_feature_cfg=vis_features_cfg,
            fold_index=fold_index,
        )
    elif model_name in ["gan", "sup_cgan"]:
        lightning_module = LightGANForecaster(
            model_name=model_name,
            data_cfg=data_cfg,
            network_cfg=network_cfg,
            hyperparameters_cfg=hyperparameters_cfg,
            visual_feature_cfg=vis_features_cfg,
            fold_index=fold_index,
        )
    elif model_name in ["vae", "sup_cvae"]:
        lightning_module = LightVAEForecaster(
            model_name=model_name,
            data_cfg=data_cfg,
            network_cfg=network_cfg,
            hyperparameters_cfg=hyperparameters_cfg,
            visual_feature_cfg=vis_features_cfg,
            fold_index=fold_index,
        )
    else:
        raise NotImplementedError(model_name)
    trainer = pl.Trainer(
        default_root_dir=save_cfg["path"],
        logger=logger,
        accelerator=accelerator,
        callbacks=callbacks,
        max_epochs=hyperparameters_cfg["max_epochs"],
        check_val_every_n_epoch=hyperparameters_cfg["val_freq"],
    )
    return trainer, lightning_module, train_dl, val_dl, val_dl
