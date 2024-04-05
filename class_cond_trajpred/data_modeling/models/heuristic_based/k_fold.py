import os
import logging
from pathlib import Path
import click
from dotenv import find_dotenv, load_dotenv
import numpy as np
import pandas as pd

from .constant_velocity_model import ConstantVelocityModel
from .eval import evaluate, evaluate_multi_label
from ....utils import load_config
from ....io import create_dir


@click.command()
@click.argument("cfg_file", type=click.Path(exists=True))
def main(cfg_file):
    logger = logging.getLogger(__name__)
    logger.info("Running k-fold with CVM")
    cfg = load_config(cfg_file, "class_cond_trajpred/cfg/heuristic_based/default.yaml")
    predictor = ConstantVelocityModel(cfg)

    data_cfg = cfg["data"]
    dataset = data_cfg["dataset"]
    obs_len = cfg["data"]["obs_len"]
    path_ds_target = (
        cfg["data"]["data_dir"]
        if dataset == "sdd"
        else os.path.join(cfg["data"]["data_dir"], cfg["data"]["dataset_target"])
    )

    overall_metrics = dict(ADE=[], FDE=[])
    metrics_per_label = dict()
    for test_index in range(1, 11):
        logger.info("Evaluating fold %d", test_index)
        test_trajectories = pd.read_pickle(
            os.path.join(path_ds_target, f"test_b{test_index}.pkl")
        )
        # metrics
        ade_res, fde_res = evaluate(
            predictor=predictor, data=test_trajectories, obs_len=obs_len
        )
        overall_metrics["ADE"].append(ade_res)
        overall_metrics["FDE"].append(fde_res)
        logger.info(
            "\nGlobal Metrics for fold %d:\n ADE=%1.2f, FDE=%1.2f",
            test_index,
            ade_res,
            fde_res,
        )
        if dataset in ["thor", "synthetic", "thor_magni", "sdd"]:
            logger.info("\n\n===Metrics per role===")
            metrics_per_label_fold = evaluate_multi_label(
                predictor=predictor,
                data=test_trajectories,
                obs_len=obs_len,
                logger=logger,
                dataset=dataset,
            )
            if len(metrics_per_label) > 0:
                for k in metrics_per_label.keys():
                    metrics_per_label[k].extend(metrics_per_label_fold[k])
            else:
                metrics_per_label = metrics_per_label_fold
        print("\n")
    logger.info("\n\n\nAvg metrics:\n")
    metrics_to_save = {}
    for metric_name, metric_val in overall_metrics.items():
        metric_mean, metric_std = np.mean(metric_val), np.std(metric_val)
        logger.info("%s=%1.2f+-%1.2f", metric_name, metric_mean, metric_std)
        metrics_to_save[metric_name] = f"{metric_mean:1.2f}+-{metric_std:1.2f}"
    if dataset in ["thor", "synthetic", "thor_magni", "sdd"]:
        logger.info("\n\n\nAvg metrics per role:\n")
        for metric_name, metric_val in metrics_per_label.items():
            metric_mean, metric_std = np.mean(metric_val), np.std(metric_val)
            logger.info("%s=%1.2f+-%1.2f", metric_name, metric_mean, metric_std)
            metrics_to_save[metric_name] = f"{metric_mean:1.2f}+-{metric_std:1.2f}"
    metrics = pd.DataFrame.from_dict(metrics_to_save, orient="index")
    save_path = os.path.join("logs", "cvm_k_fold", dataset)
    create_dir(save_path)
    metrics.to_csv(os.path.join(save_path, "results.csv"))


if __name__ == "__main__":
    LOGO_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOGO_FMT)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
