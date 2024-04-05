from typing import Dict, Union, List
import torch
import numpy as np
import pandas as pd
from collections import defaultdict


def get_forecasting_input_data(
    train_batch: dict, obs_len: int, inputs: Union[str, List[str]]
) -> Dict:
    """Split trajectory info for trajectory forecaster"""
    inputs = [inputs] if isinstance(inputs, str) else inputs
    idx_st = (
        {k: 0 for k in train_batch.keys()}
        if len(inputs) > 1
        else {k: 0 if k == "trajectories" else 1 for k in train_batch.keys()}
    )
    observed_tracklet, gt_tracklet = {}, {}
    for input_type, input_data in train_batch.items():
        if input_type == "img":
            observed_tracklet[input_type] = input_data
            gt_tracklet[input_type] = input_data
            continue
        observed_tracklet[input_type] = input_data[:, idx_st[input_type]: obs_len, ...]
        gt_tracklet[input_type] = input_data[:, obs_len:, ...]
    return observed_tracklet, gt_tracklet


class PixWorldConverter:
    """Pixel to world converter"""

    def __init__(self, info: dict) -> None:
        self.resolution = info["resolution_pm"]  # 1pix -> m
        self.offset = np.array(info["offset"])

    def convert2pixels(
        self, world_locations: Union[np.array, torch.Tensor]
    ) -> Union[np.array, torch.Tensor]:
        if world_locations.ndim == 2:
            return (world_locations / self.resolution) - self.offset

        new_world_locations = [
            self.convert2pixels(world_location) for world_location in world_locations
        ]
        return (
            torch.stack(new_world_locations)
            if isinstance(world_locations, torch.Tensor)
            else np.stack(new_world_locations)
        )

    def convert2world(
        self, pix_locations: Union[np.array, torch.Tensor]
    ) -> Union[np.array, torch.Tensor]:
        return (pix_locations + self.offset) * self.resolution


def get_thor_mapping_roles(trajs_concat: pd.DataFrame, collapse_visitors: bool):
    """return mapping between roles and int"""

    roles = trajs_concat.data_label.unique().tolist()
    mapping = defaultdict(lambda: len(roles))  # if unknown -> new class test set
    mapping.update({role: i for i, role in enumerate(roles)})
    if not collapse_visitors:
        return mapping

    # group visitors in the same label
    first_occurence, new_mapping = None, {}
    for role, numerical_label in mapping.items():
        if role.split("_")[0] == "visitors":
            if first_occurence is not None:
                new_mapping[role] = first_occurence
            else:
                first_occurence = numerical_label
                new_mapping[role] = first_occurence

        else:
            new_mapping[role] = (
                max(-1, max(new_mapping.values())) + 1
                if len(new_mapping.values()) > 0
                else 0
            )
    return new_mapping


def get_sdd_mapping_roles(trajs_concat: pd.DataFrame):
    """return mapping between roles and int"""

    roles = trajs_concat.data_label.unique().tolist()
    mapping = defaultdict(lambda: len(roles))  # if unknown -> new class test set
    mapping.update({role: i for i, role in enumerate(roles)})
    return mapping
