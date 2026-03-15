import torch
from tqdm import tqdm
import torch.nn.functional as F
import scipy.spatial.distance as ssd
import numpy as np
from scipy.spatial.distance import cdist

def get_device():
    """
    Choose the most capable computation device available: CUDA, MPS (Mac GPU), or CPU.
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def mira(truth: torch.Tensor,
          posterior: torch.Tensor,
          num_runs: int = 100,
          norm: bool = False,
          center_choice: torch.Tensor = None,
          reference_choice: torch.Tensor = None,
          disable_tqdm: bool = False,
          device: torch.device = None
          ) -> (torch.Tensor, torch.Tensor):
    """
    Monte Carlo estimation of predictive probabilities, calibration, and per-model normalized counts.
    Normalizes all inputs via Z-score (global mean/std across truth and posterior).

    Parameters
    ----------
    truth : Tensor of shape (T, q)
        Ground-truth parameter vectors (T samples in q dims).
    posterior : Tensor of shape (M, T, S, q)
        Posterior draws from M models, T truths, S samples each in q dims.
    num_runs : int
        Number of Monte Carlo replications.
    device : torch.device, optional
        Computation device; auto-detected if None.

    Returns
    -------
    score : Tensor of shape (M,)
        Mira Score per model across runs.
    pkn_values : Tensor of shape (num_runs, M, T)
        Probability of truth being in region per run/model/truth.
    """
    # Device setup
    device = device or get_device()
    truth = truth.to(device)
    posterior = posterior.to(device)

    # Shapes
    M, T, S, q = posterior.shape
    if truth.shape != (T, q):
        raise ValueError(f"Expected truth shape {(T, q)}, got {tuple(truth.shape)}")

    # -------------------------
    # Z-score normalization
    # -------------------------
    if norm:
        min_val = truth.min(dim=0, keepdim=True).values
        max_val = truth.max(dim=0, keepdim=True).values
        range_val = max_val - min_val + 1e-8  # avoid divide by zero

        truth = (truth - min_val) / range_val
        posterior = (posterior - min_val) / range_val
    # Constants
    N = S - 1
    max_val = (N + 1) / (N + 2)

    # Pre-allocate
    total_score = torch.zeros((num_runs, M), device=device)
    if disable_tqdm:
        tqdm_iter = range(num_runs)
    else:
        tqdm_iter = tqdm(range(num_runs), desc="Mira MC runs")
    for run in tqdm_iter:
        # 1. Random centers ~ U[0,1]^q
        if center_choice is not None:
            centers = center_choice # Samples from the prior to define the centers
        else:
            centers = torch.rand((T, q), device=device) # Default way

        # 2. Distances (M, T, S)
        dists = torch.norm(centers[None, :, None, :] - posterior, dim=3)

        # 3. Random radius per (M, T)
        if reference_choice is not None:
            # Samples from the prior to define the references
            ref_dists = torch.norm(centers - reference_choice, dim=1)  # (T,)
            radii = ref_dists.unsqueeze(0).expand(M, -1) + 1e-12       # (M, T)
        else:
            rand_idx = torch.randint(0, S, (M, T), device=device)
            m_idx = torch.arange(M, device=device)[:, None]
            t_idx = torch.arange(T, device=device)[None, :]
            radii = dists[m_idx, t_idx, rand_idx] # + 1e-12
            mask = torch.ones((M, T, S), dtype=torch.bool, device=device)
            mask[m_idx, t_idx, rand_idx] = False
            masked_dists = dists.masked_fill(~mask, float('inf'))


        # 4. Truth distances
        true_dists = torch.norm(centers - truth, dim=1)       # (T,)
        k = (true_dists[None, :] <= radii).float()            # (M, T)

        # 5. Counts per radius
        counts = (masked_dists < radii.unsqueeze(2)).sum(dim=2)

        # 6. Predictive probability
        prob_in = (counts + 1) / (N + 2)
        prob_out = (N - counts + 1) / (N + 2)
        prob = prob_in * k + prob_out * (1 - k)

        # 7. Calibration
        calib = prob / max_val

        # 8. Aggregate
        total_score[run] = calib.mean(dim=1)

    # Aggregate across runs
    mean_score = total_score.mean(dim=0)           # (M,)
    std_score = total_score.std(dim=0, unbiased=True)  # (M,)

    return mean_score, std_score

def mira_bootstrap(truth: torch.Tensor,
                    posterior: torch.Tensor,
                    num_bootstrap: int = 100,
                    num_runs: int = 100,
                    norm: bool = False,
                    disable_tqdm: bool = False,
                    device: torch.device = None
                    ) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    """
    Bootstrap wrapper producing per-bootstrap, per-run, per-model n/N arrays.

    Returns
    -------
    boot_score : Tensor of shape (num_bootstrap, M)
    """
    device = device or get_device()
    truth = truth.to(device)
    posterior = posterior.to(device)

    M, T, S, q = posterior.shape
    boot_score = torch.zeros((num_bootstrap, M), device=device)

    for b in tqdm(range(num_bootstrap), desc="Bootstrapping mira"):
        # Resample
        idx = torch.randint(0, T, (T,), device=device)
        truth_bs = truth[idx]
        posterior_bs = posterior[:, idx, :, :]

        # Run mira
        mira_bootstrap_score, _ = mira(
            truth_bs,
            posterior_bs,
            num_runs=1,
            norm=norm,
            disable_tqdm=disable_tqdm,
            device=device
        )
        boot_score[b] = mira_bootstrap_score

    # Mean and std across bootstraps
    boot_mean = boot_score.mean(dim=0)              # (M,)
    boot_std = boot_score.std(dim=0, unbiased=True) # (M,)

    return boot_mean, boot_std