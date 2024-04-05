import os
import logging
from pathlib import Path
import click
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.signal.windows import gaussian

from .eval import evaluate, evaluate_multi_label
from ....utils import load_config


class ConstantVelocityModel:
    def __init__(self, cfg) -> None:
        self.v0_mode = cfg["param"]["meta"]["v0_mode"]
        self.v0_sigma = cfg["param"]["meta"]["v0_sigma"]
        self.pred_len = cfg["data"]["pred_len"]

    def predict_dataset(self, dataset):
        prediction, position, v_mean = [], [], []
        for n_t, trajectory in tqdm(enumerate(dataset)):
            last_position = trajectory.iloc[-1][["x", "y"]]
            mean_velocity = self.get_weighted_displacement(trajectory)
            position.append(last_position)
            v_mean.append(mean_velocity)
        v_mean = np.squeeze(v_mean)
        last_position = np.array(position)
        for _ in range(self.pred_len):
            new_position = last_position + v_mean
            prediction.append(new_position)
            last_position = new_position
        return np.stack(prediction, axis=1).astype(float)

    def get_weighted_displacement(self, trajectory):
        period = trajectory.index.to_series().diff()[1:].median()
        displacements = trajectory[["x_speed", "y_speed"]].values[1:, :] * period
        weights = np.expand_dims(self.get_w(displacements), axis=0)
        new_displacements = np.dot(weights, displacements)
        return new_displacements

    def get_w(self, velocity):
        velocity_len = len(velocity)
        if self.v0_mode == "linear":
            weights = np.ones(velocity_len) / velocity_len
        elif self.v0_mode == "gaussian":
            window = gaussian(2 * velocity_len, self.v0_sigma)
            w1 = window[0:velocity_len]
            scale = np.sum(w1)
            # scale = np.linalg.norm(w1)
            weights = w1 / scale
            # print(w)
        elif self.v0_mode == "constant":
            weights = np.zeros(velocity_len)
            weights[-1] = 1
        else:
            raise NotImplementedError(self.v0_mode)
        return weights


@click.command()
@click.argument("cfg_file", type=click.Path(exists=True))
def main(cfg_file):
    logger = logging.getLogger(__name__)
    logger.info("Predicting with CVM")
    cfg = load_config(cfg_file, "class_cond_trajpred/cfg/heuristic_based/default.yaml")
    predictor = ConstantVelocityModel(cfg)

    data_cfg = cfg["data"]
    dataset = data_cfg["dataset"]
    obs_len = cfg["data"]["obs_len"]
    if dataset != "sdd":
        fp = os.path.join(
            data_cfg["data_dir"],
            data_cfg["dataset_target"],
            "test.pkl",
        )
    else:
        fp = os.path.join(data_cfg["data_dir"], "test.pkl")

    # data
    data = pd.read_pickle(fp)

    # metrics
    ade_res, fde_res = evaluate(predictor=predictor, data=data, obs_len=obs_len)
    logger.info("\n\nGlobal Metrics:\n ADE=%1.2f, FDE=%1.2f", ade_res, fde_res)
    if dataset in ["thor", "synthetic", "thor_magni", "sdd"]:
        logger.info("\n\n===Metrics per role===")
        evaluate_multi_label(
            predictor=predictor,
            data=data,
            obs_len=obs_len,
            logger=logger,
            dataset=dataset,
        )


if __name__ == "__main__":
    LOGO_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOGO_FMT)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
