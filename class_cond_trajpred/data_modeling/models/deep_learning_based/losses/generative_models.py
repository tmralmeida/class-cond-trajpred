from copy import deepcopy
from typing import Tuple, Callable
import torch
import torch.nn.functional as F


def variety_loss(
    n_samples: int,
    obs_tracklet_data: dict,
    y_gt: dict,
    model,
    scaler,
    get_unscaled_outputs: Callable[[torch.Tensor, dict], torch.Tensor],
    loss: Callable,
    loss_weight: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Compute k-variety loss proposed by Social GAN: https://arxiv.org/pdf/1803.10892.pdf

    Parameters
    ----------
    n_samples
        number of samples to generate (k)
    obs_tracklet_data
        raw input data -> observed tracklet data
    y_gt
        ground truth data
    model
        pytorch nn.Module
    scaler
        class object that scales the input data
    get_unscale_outputs
        function to unscale the outputs
    loss
        callable loss function
    loss_weight
        weight to apply to the L2 loss function

    Returns
    -------
        Top-1 generated tracklet, Top-3 generated tracklet, loss, model stats if vae
    """
    model_training = model.training
    model_input = scaler.scale_inputs(obs_tracklet_data)
    k_generated_tracklets, g_traj_loss, cvae_stats = [], [], []
    for _ in range(n_samples):
        generated_tracklets = model(
            model_input,
            y=scaler.scale_inputs(deepcopy(y_gt)) if model_training else None,
            scaler=scaler,
        )
        if isinstance(generated_tracklets, tuple):
            if model_training:
                cvae_stats.append((generated_tracklets[1], generated_tracklets[2]))
            generated_tracklets = generated_tracklets[0]
        y_hat = get_unscaled_outputs(generated_tracklets.clone(), obs_tracklet_data)
        k_generated_tracklets.append(generated_tracklets.clone())
        k_loss = loss(y_hat, y_gt["trajectories"])
        if k_loss.dim() != 1:
            k_loss = k_loss.mean(dim=-1).mean(dim=-1)
        g_traj_loss.append(loss_weight * k_loss)
    g_traj_loss = torch.stack(g_traj_loss, dim=1)
    k_generated_tracklets = torch.stack(k_generated_tracklets, dim=0)
    k_loss, min_index = torch.min(g_traj_loss, dim=1)
    avg_k_loss = k_loss.mean()
    best_generated_tracklets = torch.zeros(k_generated_tracklets.shape[1:]).to(
        k_generated_tracklets
    )
    best_model_stats = None
    if len(cvae_stats) > 0 and model_training:
        best_model_stats = (
            dict(
                mu=torch.zeros_like(cvae_stats[0][0]),
                log_var=torch.zeros_like(cvae_stats[0][1]),
            )
            if len(cvae_stats) > 0
            else None
        )
    for i in range(len(min_index)):
        best_generated_tracklets[i] = k_generated_tracklets[min_index[i], i, :, :]
        if len(cvae_stats) > 0 and model_training:
            best_model_stats["mu"][i] = cvae_stats[min_index[i]][0][i]
            best_model_stats["log_var"][i] = cvae_stats[min_index[i]][1][i]

    return (
        k_generated_tracklets[0],
        best_generated_tracklets,
        avg_k_loss,
        best_model_stats,
    )


def feature_matching_loss(discriminator, **kwargs) -> torch.Tensor:
    """Feature-mathcing loss used as an alternative to bce loss on the GAN's generator"""
    gt_tracklets = kwargs["gt_tracklets"]
    generated_tracklets = kwargs["tracklets"]
    feature_real = discriminator(
        gt_tracklets, get_features=True, labels=kwargs["labels"]
    )
    feature_fake = discriminator(
        generated_tracklets, get_features=True, labels=kwargs["labels"]
    )
    return F.mse_loss(feature_fake, feature_real.detach())


def bce_loss(discriminator, **kwargs) -> torch.Tensor:
    """BCE loss used in GANs training"""
    tracklets = kwargs["tracklets"]
    gt_labels = kwargs["gt_labels"].to(tracklets)
    disc_scores = discriminator(tracklets, labels=kwargs["labels"])
    return F.binary_cross_entropy_with_logits(disc_scores, gt_labels)


def kl_divergence_loss(mu: torch.Tensor, log_var: torch.Tensor, weight: float = 1.0):
    """kl divergence loss"""
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), axis=-1)
    return kl_loss.sum() * weight
