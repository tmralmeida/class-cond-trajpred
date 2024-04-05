import math
import torch
from torchmetrics import Metric


class TopkAdeFde(Metric):
    def __init__(self, top_k: int, weight_ade: float = 0.5, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("ade", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fde", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.top_k = top_k
        self.w_ade = weight_ade

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        if preds.shape[0] != self.top_k:
            raise ValueError("Predictions must be (top_k, bs, ts, 2)")
        losses = {"ade": [], "fde": [], "wavg": []}
        for _pred in preds:
            last_gt, last = target[:, -1, :], _pred[:, -1, :]
            fde_loss = torch.norm(last_gt - last, 2, 1)
            ade_loss = torch.mean(torch.norm(preds - target, dim=-1), dim=-1)
            losses["ade"].append(ade_loss)
            losses["fde"].append(fde_loss)
            losses["wavg"].append(ade_loss * self.w_ade + (1 - self.w_ade) * fde_loss)
        for metric_name, vals in losses.items():
            losses[metric_name] = torch.stack(vals, dim=-1)
        idx_min = torch.argmin(losses["wavg"], dim=-1, keepdim=True)
        self.ade += torch.gather(losses["ade"], 1, idx_min).sum()
        self.fde += torch.gather(losses["fde"], 1, idx_min).sum()
        self.total += target.shape[0]  # evaluating bs

    def compute(self):
        return {
            f"top-{self.top_k} ADE": self.ade.float() / self.total,
            f"top-{self.top_k} FDE": torch.sqrt(self.fde.float() / self.total),
        }


class NegativeCondLogLikelihood(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("cll", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        if preds.shape != target.shape:
            raise ValueError("Problem on CLL shapes")

        rdiffs = 0.5 * torch.square(preds - target).sum(dim=-1).sum(dim=-1)
        rexp = (1.0 / math.sqrt(2 * math.pi)) * torch.exp(-rdiffs)
        rmean = rexp.mean(dim=1).clamp(min=1e-12)
        self.cll += -torch.log(rmean).sum()
        self.total += target.shape[0]  # evaluating bs

    def compute(self):
        return self.cll.float() / self.total


def compute_cll(
    t_cll: int,
    generator,
    obs_tracklet_data: dict,
    y_gt: torch.Tensor,
    scaler,
    get_unscaled_outputs: callable,
) -> dict:
    # compute negative conditional log likelihood
    cll_out = dict(cll_y_hat=[], cll_gt=[])
    with torch.no_grad():
        for _ in range(t_cll):
            generated_tracklets = generator(
                scaler.scale_inputs(obs_tracklet_data), y=None, scaler=scaler
            )
            if isinstance(generated_tracklets, tuple):
                generated_tracklets = generated_tracklets[0]
            y_hats = get_unscaled_outputs(
                generated_tracklets.clone(), obs_tracklet_data
            )
            cll_out["cll_y_hat"].append(y_hats)
            cll_out["cll_gt"].append(y_gt)
        cll_out["cll_y_hat"] = torch.stack(cll_out["cll_y_hat"], dim=1)
        cll_out["cll_gt"] = torch.stack(cll_out["cll_gt"], dim=1)
    return cll_out
