import torch


def rmse(Y: torch.Tensor, MU: torch.Tensor, return_mean: bool = True) -> torch.Tensor:
    if Y.ndim == 1:
        Y = Y.unsqueeze(1)
    if MU.ndim == 1:
        MU = MU.unsqueeze(1)
    Z = torch.sqrt(torch.mean((Y - MU) ** 2, dim=0))
    if return_mean:
        return torch.mean(Z)
    return Z


def crps(Y, MU, SIGMA, return_mean=True):
    if Y.ndim == 1:
        Y = Y.unsqueeze(1)
    if MU.ndim == 1:
        MU = MU.unsqueeze(1)

    SIGMA = SIGMA.unsqueeze(1)  # Shape (N, 1)
    Z = (Y - MU) / SIGMA  # Shape (N, M)

    # Standard normal PDF and CDF
    pdf = torch.exp(-0.5 * Z**2) / torch.sqrt(torch.tensor(2 * torch.tensor(torch.pi)))
    cdf = 0.5 * (1 + torch.erf(Z / torch.sqrt(torch.tensor(2.0))))

    # Compute CRPS
    crps = SIGMA * (
        Z * (2 * cdf - 1) + 2 * pdf - 1 / torch.sqrt(torch.tensor(torch.pi))
    )

    if return_mean:
        return torch.mean(torch.mean(crps, dim=0))

    return crps


def coverage(
    Y: torch.Tensor, MU: torch.Tensor, SIGMA: torch.Tensor, return_mean: bool = True
) -> torch.Tensor:
    if Y.ndim == 1:
        Y = Y.unsqueeze(1)
    if MU.ndim == 1:
        MU = MU.unsqueeze(1)
    SIGMA = SIGMA.unsqueeze(1)

    lower = MU - 1.96 * SIGMA
    upper = MU + 1.96 * SIGMA

    coverage = ((Y > lower) & (Y < upper)).float()
    coverage_prob = torch.mean(coverage, dim=0)

    if return_mean:
        return torch.mean(coverage_prob)
    return coverage_prob
