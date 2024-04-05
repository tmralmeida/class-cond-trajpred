# -*- coding: utf-8 -*-
import os
import logging
from pathlib import Path
import click
from dotenv import find_dotenv, load_dotenv
import ray
import numpy as np
import pandas as pd

from .utils import get_trainer_objects_kfold_cv_from_yufei, run_trainer
from ..utils import load_config, merge_dicts
from ..io import load_json_file, dump_json_file


@ray.remote
def ray_run_trainer(trainer_options):
    return run_trainer(trainer_options)


@click.command()
@click.argument("k_folds", type=click.INT)
@click.argument("cfg_file", type=click.Path(exists=True))
def main(k_folds, cfg_file):
    """Runs model training n times"""
    DEFAULT_PATH = "class_cond_trajpred/cfg/default.yaml"
    logger = logging.getLogger(__name__)
    cfg = load_config(cfg_file, DEFAULT_PATH)
    data_cfg = cfg["data"]
    dataset_name = data_cfg["dataset"]
    accelerator = "gpu" if cfg["visual_feature_extractor"]["use"] else "cpu"
    model_name = cfg["model"]
    dataset_target = (
        ""
        if dataset_name == "sdd"
        else os.path.join(cfg["data"]["data_dir"], cfg["data"]["dataset_target"])
    )
    logger.info("Training %s", "-".join([dataset_name, dataset_target, model_name]))

    if accelerator == "cpu":
        ray.init()
        trainers = [
            get_trainer_objects_kfold_cv_from_yufei(
                data_cfg["data_dir"], i, cfg, accelerator, f"fold_{i}"
            )
            for i in range(1, k_folds + 1)
        ]
        logging_paths = ray.get(
            [ray_run_trainer.remote(tr_options) for tr_options in trainers]
        )
    else:
        raise ValueError(accelerator)

    save_path = os.path.join(os.path.dirname(logging_paths[0]), "n_runs_metrics.json")
    overall_metrics = load_json_file(
        os.path.join(logging_paths[0], "test_metrics.json")
    )
    for logging_path in logging_paths[1:]:
        test_metrics = load_json_file(os.path.join(logging_path, "test_metrics.json"))
        overall_metrics = merge_dicts(overall_metrics, test_metrics)
    dump_json_file(overall_metrics, save_path)

    logger.info("==================Overall Results================")
    logger.info(overall_metrics)

    metrics_df = {}
    for k, v in overall_metrics.items():
        if k != "labels_mapping":
            avg_metric, std_metric = np.mean(v), np.std(v)
            logger.info("%s %1.2f +- %1.2f", k, avg_metric, std_metric)
            metrics_df[k] = f"{avg_metric:1.2f}+-{std_metric:1.2f}"

    metrics = pd.DataFrame.from_dict(metrics_df, orient="index")
    metrics.to_csv(os.path.join(os.path.dirname(logging_paths[0]), "average_runs.csv"))


if __name__ == "__main__":
    LOGO_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOGO_FMT)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
