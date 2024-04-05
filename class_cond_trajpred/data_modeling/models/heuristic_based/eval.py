from typing import List
import pandas as pd
import numpy as np
import torch

from ....evaluation.common_metrics import (
    FinalDisplacementError,
    AverageDisplacementError,
)


def evaluate(predictor, data: List[pd.DataFrame], obs_len: int):
    """Evaluate dataset"""
    ade = AverageDisplacementError()
    fde = FinalDisplacementError()
    obs_dataset = list(map(lambda x: x.iloc[:obs_len], data))
    gt_dataset = list(map(lambda x: x.iloc[obs_len:][["x", "y"]].values, data))

    y_hat = predictor.predict_dataset(obs_dataset)
    y_true = np.stack(gt_dataset).astype(float)

    ade.update(preds=torch.from_numpy(y_hat), target=torch.from_numpy(y_true))
    fde.update(preds=torch.from_numpy(y_hat), target=torch.from_numpy(y_true))
    ade_res = ade.compute().item()
    fde_res = fde.compute().item()
    ade.reset()
    fde.reset()
    return ade_res, fde_res


def evaluate_multi_label(
    predictor, data: List[pd.DataFrame], obs_len: int, logger, dataset: str
):
    """Evaluate dataset per existing label"""
    sup_labels = pd.concat(data)["data_label"].unique()
    if dataset == "thor":
        sup_labels = set(
            [
                sup_label if "visitors" not in sup_label else "visitors"
                for sup_label in sup_labels
            ]
        )
    res_metrics = {}
    for sup_label in sup_labels:
        target_data = list(
            filter(
                lambda x: str(sup_label) in str(x["data_label"].iloc[0])
                if dataset == "thor"
                else str(sup_label) == str(x["data_label"].iloc[0]),
                data,
            )
        )
        ade_res, fde_res = evaluate(predictor, target_data, obs_len)
        logger.info("[%s] Metrics:\n ADE=%1.2f, FDE=%1.2f", sup_label, ade_res, fde_res)
        res_metrics["ADE_" + sup_label] = [ade_res]
        res_metrics["FDE_" + sup_label] = [fde_res]
    return res_metrics
