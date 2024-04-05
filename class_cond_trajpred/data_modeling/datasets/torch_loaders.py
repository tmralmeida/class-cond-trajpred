import os
from typing import List, Optional
from PIL import Image
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

from class_cond_trajpred.io import load_json_file
from .utils import PixWorldConverter, get_thor_mapping_roles, get_sdd_mapping_roles


def load_dataset(set_type: str, cfg: dict, **kwargs):
    """Loading dataset according to the mode (train, val, or test) and the config file
    Args:
        set_type (str): `train`, `val`, or `test`
        cfg (dict): config file
    Raises:
        NotImplementedError: if the dataset name is not accepted
    Returns:
        [type]: dataset object
    """
    dataset = cfg["dataset"]
    tested_ds = cfg["dataset_target"]
    _path = (
        os.path.join(cfg["data_dir"], tested_ds)
        if tested_ds is not None
        else cfg["data_dir"]
    )
    trajectories = DatasetFromPath.get_data(trajectories_path=_path, set_type=set_type)

    if dataset == "eth_ucy":
        return DeepLearningDataset(trajectories)
    elif dataset in ["thor", "thor_magni"]:
        visuals_path, window_size, obs_len = None, None, None
        if "visuals_path" in cfg:
            visuals_path, window_size, obs_len = (
                cfg["visuals_path"],
                cfg["window_size"],
                cfg["observation_len"],
            )
        return ThorDataset(
            trajectories=trajectories,
            visuals_path=visuals_path,
            window_size=window_size,
            obs_len=obs_len,
            collapse_visitors=cfg["collapse_visitors"],
        )
    elif dataset == "synthetic":
        if tested_ds == "three_modes":
            mapping_modes = dict(straight=0, ascending=1, descending=2)
        return SyntheticDataset(
            trajectories_path=_path, set_type=set_type, mapping_modes=mapping_modes
        )
    elif dataset == "sdd":
        return SDDDataset(trajectories=trajectories)
    else:
        raise NotImplementedError(dataset)


class DatasetFromPath:
    """Load trajectories form input path object"""

    @staticmethod
    def get_data(trajectories_path: str, set_type: str) -> None:
        if set_type not in ["train", "val", "test"]:
            raise NameError(f"{set_type} is not in ['train', 'val', 'test']!")
        return pd.read_pickle(os.path.join(trajectories_path, set_type + ".pkl"))


class DeepLearningDataset(Dataset):
    """Default dataset loader object"""

    def __init__(self, trajectories: List[pd.DataFrame]) -> None:
        super().__init__()
        self.input_data = trajectories

    def convert_to_torch(self, arr: np.array) -> torch.Tensor:
        return torch.from_numpy(arr).type(torch.float)

    def get_common_inputs(self, index):
        target_data = self.input_data[index]
        locations = self.convert_to_torch(target_data[["x", "y"]].values)
        displacements = self.convert_to_torch(
            target_data[["x_delta", "y_delta"]].values
        )
        polars = self.convert_to_torch(target_data[["n_deltas", "theta_delta"]].values)
        speed = self.convert_to_torch(target_data[["x_speed", "y_speed"]].values)
        # straightness_index = self.convert_to_torch(
        #     target_data["straightness_index"].values
        # )
        period = (
            target_data.time.diff()
            if "time" in target_data.columns
            else target_data.index.to_series().diff()
        )
        period = period.fillna(period.median())
        period = self.convert_to_torch(period.values)
        return dict(
            trajectories=locations,
            displacements=displacements,
            speeds=speed,
            polars=polars,
            # straightness_index=straightness_index,
            period=period,
        )

    def __getitem__(self, index):
        return self.get_common_inputs(index)

    def __len__(self):
        return len(self.input_data)


class ThorDataset(DeepLearningDataset):
    def __init__(
        self,
        trajectories: List[pd.DataFrame],
        visuals_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(trajectories)
        trajs_concat = pd.concat(self.input_data)
        self.mapping_roles = get_thor_mapping_roles(
            trajs_concat, collapse_visitors=kwargs["collapse_visitors"]
        )
        self.mapping_datasets = dict(
            Scenario_1=0, Scenario_2=1, Scenario_3=2, Scenario_3A=3, Scenario_3B=4
        )
        self.imgs, self.window_size, self.pix2world_conv = {}, None, None
        if visuals_path:
            visuals_info = load_json_file(os.path.join(visuals_path, "info.json"))
            self.obs_len = kwargs["obs_len"]
            self.pix2world_conv = PixWorldConverter(visuals_info)
            self.window_size = int(
                kwargs["window_size"] / self.pix2world_conv.resolution
            )
            datasets = trajs_concat.dataset_name.unique()
            for scenario in datasets:
                vis_path = os.path.join(visuals_path, scenario + ".png")
                img = np.array(Image.open(vis_path))
                self.imgs[scenario] = np.flipud(img[:, :, :3])

    def get_mapping_cat_vars(self, cat_vars: List[str], mapping: dict):
        return self.convert_to_torch(
            np.array([mapping[cat_var] for cat_var in cat_vars])
        )

    def __getitem__(self, index):
        new_inputs = self.get_common_inputs(index)
        roles = self.input_data[index]["data_label"].values
        datasets = self.input_data[index]["dataset_name"].values
        new_inputs.update(
            {
                "data_label": self.get_mapping_cat_vars(roles, self.mapping_roles),
                "dataset": self.get_mapping_cat_vars(datasets, self.mapping_datasets),
            }
        )
        if len(self.imgs) > 0:
            dataset = self.input_data[index]["dataset_name"].iloc[0]
            visual = torch.from_numpy(
                self.imgs[dataset].transpose(2, 0, 1).copy()
            ).float()
            visual /= visual.max()
            trajs_pix = self.pix2world_conv.convert2pixels(new_inputs["trajectories"])
            last_pt = trajs_pix[self.obs_len - 1]
            col_min, col_max = (
                max(0, int(last_pt[0]) - self.window_size),
                min(int(last_pt[0]) + self.window_size, visual.shape[1] - 1),
            )
            row_min, row_max = (
                max(0, int(last_pt[1]) - self.window_size),
                min(int(last_pt[1]) + self.window_size, visual.shape[2] - 1),
            )
            target_visual = visual[:, row_min:row_max, col_min:col_max]
            new_inputs.update(dict(img=target_visual))
        return new_inputs


class SyntheticDataset(DeepLearningDataset):
    def __init__(self, trajectories, **kwargs) -> None:
        super().__init__(trajectories)
        self.mapping_modes = kwargs["mapping_modes"]

    def __getitem__(self, index):
        new_inputs = self.get_common_inputs(index)
        roles = self.input_data[index]["data_label"].values
        new_inputs.update(dict(data_label=roles))
        return new_inputs


class SDDDataset(DeepLearningDataset):
    def __init__(self, trajectories: List[pd.DataFrame]) -> None:
        super().__init__(trajectories)
        trajs_concat = pd.concat(self.input_data)
        self.mapping_roles = get_sdd_mapping_roles(trajs_concat)

    def get_mapping_cat_vars(self, cat_vars: List[str], mapping: dict):
        return self.convert_to_torch(
            np.array([mapping[cat_var] for cat_var in cat_vars])
        )

    def __getitem__(self, index):
        new_inputs = self.get_common_inputs(index)
        roles = self.input_data[index]["data_label"].values
        new_inputs.update(
            {
                "data_label": self.get_mapping_cat_vars(roles, self.mapping_roles),
            }
        )
        return new_inputs
