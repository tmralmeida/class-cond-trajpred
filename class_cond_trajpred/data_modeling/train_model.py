# -*- coding: utf-8 -*-
import logging
from pathlib import Path
import click
from dotenv import find_dotenv, load_dotenv

from .utils import get_trainer_objects
from ..utils import load_config


@click.command()
@click.argument("cfg_file", type=click.Path(exists=True))
def main(cfg_file):
    """Runs model training"""
    DEFAULT_PATH = "class_cond_trajpred/cfg/default.yaml"
    logger = logging.getLogger(__name__)
    cfg = load_config(cfg_file, DEFAULT_PATH)
    data_cfg = cfg["data"]
    dataset_name = data_cfg["dataset"]
    dataset_target = data_cfg["dataset_target"]
    model_name = cfg["model"]
    accelerator = "gpu" if cfg["visual_feature_extractor"]["use"] else "cpu"
    str_logs = (
        [dataset_name, dataset_target, model_name]
        if dataset_target
        else [dataset_name, model_name]
    )
    logger.info("Training %s", "-".join(str_logs))
    trainer, lightning_module, train_dl, val_dl, test_dl = get_trainer_objects(
        cfg, accelerator
    )
    trainer.fit(lightning_module, train_dl, val_dl)
    if test_dl:
        trainer.test(ckpt_path="best", dataloaders=test_dl)


if __name__ == "__main__":
    LOGO_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOGO_FMT)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
