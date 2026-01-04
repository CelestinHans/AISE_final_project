import torch

def sample_from_domain_points(domain_pos: torch.Tensor, num_tokens: int, seed: int | None = None) -> torch.Tensor:
    """
    domain_pos: [M, 2] (points du domaine D)
    returns: latent_pos [N, 2] tokens latents
    """
    M = domain_pos.shape[0]
    if num_tokens >= M:
        return domain_pos

    g = None
    if seed is not None:
        g = torch.Generator(device=domain_pos.device)
        g.manual_seed(int(seed))

    idx = torch.randperm(M, generator=g, device=domain_pos.device)[:num_tokens]
    return domain_pos[idx]
