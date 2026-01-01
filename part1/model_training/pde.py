import torch


def laplacian(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Compute the Laplacian of a scalar field u(x) with respect to x in R^2.

    Args:
        u: Tensor of shape (n, 1).
        x: Tensor of shape (n, 2) with requires_grad=True.

    Returns:
        Laplacian tensor of shape (n, 1).
    """
    grad_u = torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True
    )[0]
    u_x = grad_u[:, 0:1]
    u_y = grad_u[:, 1:2]

    u_xx = torch.autograd.grad(
        u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True, retain_graph=True
    )[0][:, 0:1]

    u_yy = torch.autograd.grad(
        u_y, x, grad_outputs=torch.ones_like(u_y), create_graph=True, retain_graph=True
    )[0][:, 1:2]

    return u_xx + u_yy
