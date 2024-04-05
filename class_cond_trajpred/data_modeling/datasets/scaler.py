import os
import torch
from ...io import load_json_file


class IOScaler:
    def __init__(self, cfg: dict, **kwargs):
        target_dataset = cfg["dataset_target"]
        if "fold_index" in kwargs.keys():
            fold_index = kwargs["fold_index"]
            stats_path = (
                os.path.join(
                    cfg["data_dir"], target_dataset, f"stats_b{fold_index}.json"
                )
                if target_dataset
                else os.path.join(cfg["data_dir"], f"stats_b{fold_index}.json")
            )
        else:
            main_dataset = cfg["dataset"]
            path_load = "data/original/"
            stats_path = (
                os.path.join(path_load, main_dataset, target_dataset, "stats.json")
                if target_dataset
                else os.path.join(path_load, main_dataset, "stats.json")
            )
        self.stats = load_json_file(stats_path)

    def _get_stats(self, input_type: str):
        _mean, _scale = self.stats[input_type]
        _mean = torch.Tensor(_mean).unsqueeze(dim=0).unsqueeze(dim=0)
        _scale = torch.Tensor(_scale).unsqueeze(dim=0).unsqueeze(dim=0)
        return _mean, _scale

    def scale_inputs(self, x: dict) -> dict:
        """scale inputs"""
        out_scaled = {}
        for input_type, input_data in x.items():
            if input_type not in self.stats:
                out_scaled[input_type] = input_data.clone()
                continue
            _mean, _scale = self._get_stats(input_type)
            input_scaled = (input_data.clone() - _mean.to(input_data)) / _scale.to(
                input_data
            )
            out_scaled[input_type] = input_scaled
        return out_scaled

    def inv_scale_outputs(self, x: torch.Tensor, out_type: str) -> torch.Tensor:
        """inverse scaling of outputs"""
        if out_type not in self.stats:
            raise ValueError(f"{out_type} not in statistics")
        _mean, _scale = self._get_stats(out_type)
        out = x.clone() * _scale.to(x) + _mean.to(x)
        return out

    def inv_transform_speeds(self, speeds: torch.Tensor, observed_tracklet_info: dict):
        """scaled speed -> unscaled speed -> displacements -> locations"""
        period, _ = observed_tracklet_info["period"].median(dim=1)
        unscaled_speeds = self.inv_scale_outputs(speeds, out_type="speeds")
        period = period.unsqueeze(dim=1).unsqueeze(dim=2).to(unscaled_speeds)
        displacements = unscaled_speeds * period
        last_observed_points = observed_tracklet_info["trajectories"][:, -1, :].to(
            displacements
        )
        displacements[:, 0, :] += last_observed_points
        return torch.cumsum(displacements, dim=1)

    def inv_transform_displacements(
        self, displacements: torch.Tensor, observed_tracklet: torch.Tensor
    ) -> torch.Tensor:
        """unscaled displacements -> displacements -> locations"""
        unscaled_displacements = self.inv_scale_outputs(
            displacements, out_type="displacements"
        )
        last_observed_point = observed_tracklet[:, -1, :]
        unscaled_displacements[:, 0, :] += last_observed_point
        return torch.cumsum(unscaled_displacements, dim=1)
