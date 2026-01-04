from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import matplotlib.pyplot as plt
from tqdm import trange

from part1.model_training.models import MLP
from part1.model_training.pde import laplacian


@dataclass(frozen=True)
class BestCfg:
    method: str  # "datadriven" or "pinn"
    K: int
    sample_idx: int
    hidden_layers: int
    width: int
    adam_lr: float
    lambda_bc: float


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_xy_grid(n: int, device: torch.device) -> torch.Tensor:
    x = torch.linspace(0.0, 1.0, n, device=device)
    y = torch.linspace(0.0, 1.0, n, device=device)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    return torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)


def boundary_mask(n: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(n * n, dtype=torch.bool, device=device)
    for i in range(n):
        mask[i * n + 0] = True
        mask[i * n + (n - 1)] = True
    for j in range(n):
        mask[0 * n + j] = True
        mask[(n - 1) * n + j] = True
    return mask


def load_sample(data_root: Path, K: int, sample_idx: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    f = np.load(data_root / f"K{K}" / f"f_{sample_idx}.npy")
    u = np.load(data_root / f"K{K}" / f"u_{sample_idx}.npy")
    n = int(u.shape[0])

    xy = make_xy_grid(n, device=device)
    f_t = torch.tensor(f.reshape(-1, 1), dtype=torch.float32, device=device)
    u_t = torch.tensor(u.reshape(-1, 1), dtype=torch.float32, device=device)
    bc = boundary_mask(n, device=device)
    return xy, f_t, u_t, bc


def rel_improvement(prev_best: float, current: float) -> float:
    if not np.isfinite(prev_best):
        return np.inf
    return (prev_best - current) / max(prev_best, 1e-12)


def train_datadriven_with_snapshots(
    model: nn.Module,
    xy: torch.Tensor,
    u_true: torch.Tensor,
    adam_lr: float,
    adam_epochs: int,
    batch_size: int,
    lbfgs_max_iter: int,
    snapshot_every: int,
    early_tol: float,
    patience: int,
    min_epochs: int,
    verbose: bool,
    print_every: int,
) -> Tuple[torch.Tensor, List[torch.Tensor], float]:
    mse = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=adam_lr)

    n = xy.shape[0]
    snapshots: List[torch.Tensor] = []
    best = float("inf")
    bad = 0

    model.train()
    for epoch in trange(adam_epochs, disable=not verbose, desc="Adam (Data)"):
        perm = torch.randperm(n, device=xy.device)
        epoch_loss = 0.0
        nb = 0

        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb = xy[idx]
            ub = u_true[idx]

            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = mse(pred, ub)
            loss.backward()
            opt.step()

            v = float(loss.item())
            epoch_loss += v
            nb += 1

        mean_epoch = epoch_loss / max(nb, 1)

        if (epoch + 1) % snapshot_every == 0:
            snapshots.append(parameters_to_vector(model.parameters()).detach().clone())

        if mean_epoch < best:
            impr = rel_improvement(best, mean_epoch)
            best = mean_epoch
            bad = 0
        else:
            impr = 0.0
            bad += 1

        if verbose and ((epoch + 1) % print_every == 0 or epoch == 0 or epoch + 1 == adam_epochs):
            print(f"[Data][Adam] epoch {epoch+1}/{adam_epochs} mean_loss={mean_epoch:.6e} best={best:.6e} rel_impr={impr:.3e} bad={bad}/{patience}")

        if (epoch + 1) >= min_epochs:
            if impr < early_tol:
                bad += 1
            else:
                bad = 0
            if bad >= patience:
                if verbose:
                    print(f"[Data][Adam] early stop at epoch {epoch+1} (tol={early_tol}, patience={patience})")
                break

    xy_full = xy
    u_full = u_true

    lbfgs = torch.optim.LBFGS(model.parameters(), max_iter=lbfgs_max_iter, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        lbfgs.zero_grad(set_to_none=True)
        pred = model(xy_full)
        loss = mse(pred, u_full)
        loss.backward()
        return loss

    with torch.enable_grad():
        lbfgs.step(closure)

    theta_star = parameters_to_vector(model.parameters()).detach().clone()

    with torch.no_grad():
        final_loss = float(mse(model(xy_full), u_full).item())

    return theta_star, snapshots, final_loss


def train_pinn_with_snapshots(
    model: nn.Module,
    xy_all: torch.Tensor,
    f_all: torch.Tensor,
    bc_mask: torch.Tensor,
    lambda_bc: float,
    adam_lr: float,
    adam_epochs: int,
    interior_batch_size: int,
    lbfgs_max_iter: int,
    snapshot_every: int,
    early_tol: float,
    patience: int,
    min_epochs: int,
    verbose: bool,
    print_every: int,
) -> Tuple[torch.Tensor, List[torch.Tensor], float]:
    mse = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=adam_lr)

    xy_int = xy_all[~bc_mask]
    f_int = f_all[~bc_mask]
    xy_bc = xy_all[bc_mask]

    n_int = xy_int.shape[0]
    snapshots: List[torch.Tensor] = []
    best = float("inf")
    bad = 0

    model.train()
    for epoch in trange(adam_epochs, disable=not verbose, desc="Adam (PINN)"):
        perm = torch.randperm(n_int, device=xy_all.device)
        epoch_loss = 0.0
        nb = 0

        for i in range(0, n_int, interior_batch_size):
            idx = perm[i : i + interior_batch_size]
            xb = xy_int[idx].detach().requires_grad_(True)
            fb = f_int[idx]

            opt.zero_grad(set_to_none=True)

            u_pred = model(xb)
            lap = laplacian(u_pred, xb)
            r = -lap - fb
            loss_pde = mse(r, torch.zeros_like(r))

            u_bc = model(xy_bc)
            loss_bc = mse(u_bc, torch.zeros_like(u_bc))

            loss = loss_pde + lambda_bc * loss_bc
            loss.backward()
            opt.step()

            v = float(loss.item())
            epoch_loss += v
            nb += 1

        mean_epoch = epoch_loss / max(nb, 1)

        if (epoch + 1) % snapshot_every == 0:
            snapshots.append(parameters_to_vector(model.parameters()).detach().clone())

        if mean_epoch < best:
            impr = rel_improvement(best, mean_epoch)
            best = mean_epoch
            bad = 0
        else:
            impr = 0.0
            bad += 1

        if verbose and ((epoch + 1) % print_every == 0 or epoch == 0 or epoch + 1 == adam_epochs):
            print(f"[PINN][Adam] epoch {epoch+1}/{adam_epochs} mean_loss={mean_epoch:.6e} best={best:.6e} rel_impr={impr:.3e} bad={bad}/{patience}")

        if (epoch + 1) >= min_epochs:
            if impr < early_tol:
                bad += 1
            else:
                bad = 0
            if bad >= patience:
                if verbose:
                    print(f"[PINN][Adam] early stop at epoch {epoch+1} (tol={early_tol}, patience={patience})")
                break

    xy_int_full = xy_int.detach().requires_grad_(True)
    f_int_full = f_int
    xy_bc_full = xy_bc

    lbfgs = torch.optim.LBFGS(model.parameters(), max_iter=lbfgs_max_iter, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        lbfgs.zero_grad(set_to_none=True)

        u_int = model(xy_int_full)
        lap = laplacian(u_int, xy_int_full)
        r = -lap - f_int_full
        loss_pde = mse(r, torch.zeros_like(r))

        u_bc = model(xy_bc_full)
        loss_bc = mse(u_bc, torch.zeros_like(u_bc))

        loss = loss_pde + lambda_bc * loss_bc
        loss.backward()
        return loss

    with torch.enable_grad():
        lbfgs.step(closure)

    theta_star = parameters_to_vector(model.parameters()).detach().clone()

    with torch.enable_grad():
        u_int = model(xy_int_full)
        lap = laplacian(u_int, xy_int_full)
        r = -lap - f_int_full
        loss_pde = mse(r, torch.zeros_like(r))
        u_bc = model(xy_bc_full)
        loss_bc = mse(u_bc, torch.zeros_like(u_bc))
        final_loss = float((loss_pde + lambda_bc * loss_bc).item())

    return theta_star, snapshots, final_loss


def pca_directions(theta_star: torch.Tensor, snapshots: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(snapshots) < 2:
        raise ValueError("Need at least 2 snapshots to compute PCA directions.")
    X = torch.stack([s - theta_star for s in snapshots], dim=0)  # (T, P)
    X = X - X.mean(dim=0, keepdim=True)
    Xn = X.detach().cpu().numpy()
    _, _, vt = np.linalg.svd(Xn, full_matrices=False)
    d1 = torch.tensor(vt[0], dtype=theta_star.dtype, device=theta_star.device)
    d2 = torch.tensor(vt[1], dtype=theta_star.dtype, device=theta_star.device)
    d1 = d1 / (d1.norm() + 1e-12)
    d2 = d2 / (d2.norm() + 1e-12)
    return d1, d2


def loss_datadriven(model: nn.Module, xy: torch.Tensor, u_true: torch.Tensor) -> float:
    with torch.no_grad():
        pred = model(xy)
        return float(torch.mean((pred - u_true) ** 2).item())


def loss_pinn_total(model: nn.Module, xy_all: torch.Tensor, f_all: torch.Tensor, bc_mask: torch.Tensor, lambda_bc: float) -> float:
    mse = nn.MSELoss()
    xy_int = xy_all[~bc_mask].detach().requires_grad_(True)
    f_int = f_all[~bc_mask]
    xy_bc = xy_all[bc_mask]

    u_int = model(xy_int)
    lap = laplacian(u_int, xy_int)
    r = -lap - f_int
    loss_pde = mse(r, torch.zeros_like(r))

    u_bc = model(xy_bc)
    loss_bc = mse(u_bc, torch.zeros_like(u_bc))

    loss = loss_pde + lambda_bc * loss_bc
    return float(loss.item())


def eval_landscape(
    model: nn.Module,
    theta_star: torch.Tensor,
    d1: torch.Tensor,
    d2: torch.Tensor,
    alphas: np.ndarray,
    betas: np.ndarray,
    method: str,
    xy: torch.Tensor,
    u_true: torch.Tensor,
    f_all: torch.Tensor,
    bc_mask: torch.Tensor,
    lambda_bc: float,
) -> np.ndarray:
    P = theta_star.numel()
    base = theta_star.view(-1)

    grid = np.zeros((len(alphas), len(betas)), dtype=np.float64)

    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            vec = base + float(a) * d1 + float(b) * d2
            vector_to_parameters(vec, model.parameters())

            if method == "datadriven":
                grid[i, j] = loss_datadriven(model, xy, u_true)
            else:
                grid[i, j] = loss_pinn_total(model, xy, f_all, bc_mask, lambda_bc)

    vector_to_parameters(base, model.parameters())
    assert parameters_to_vector(model.parameters()).numel() == P
    return grid


def plot_contour(
    alphas: np.ndarray,
    betas: np.ndarray,
    losses: np.ndarray,
    out_png: Path,
    title: str,
    log_eps: float = 1e-12,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    A, B = np.meshgrid(betas, alphas)  # note: columns=betas, rows=alphas
    Z = np.log10(losses + log_eps)

    fig = plt.figure(figsize=(6, 5), dpi=150)
    ax = fig.add_subplot(111)
    cf = ax.contourf(B, A, Z, levels=50)
    ax.contour(B, A, Z, levels=15, linewidths=0.6)
    ax.scatter([0.0], [0.0], marker="x", s=60)
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\alpha$")
    ax.set_title(title)
    fig.colorbar(cf, ax=ax, label=r"$\log_{10}(\mathcal{L})$")
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def best_configs() -> List[BestCfg]:
    return [
        BestCfg("datadriven", 1, 0, 3, 128, 1e-3, 10.0),
        BestCfg("datadriven", 4, 0, 4, 64, 5e-4, 10.0),
        BestCfg("datadriven", 16, 0, 4, 32, 5e-4, 10.0),
        BestCfg("pinn", 1, 0, 3, 32, 5e-4, 100.0),
        BestCfg("pinn", 4, 0, 4, 32, 5e-4, 100.0),
        BestCfg("pinn", 16, 0, 3, 32, 5e-4, 100.0),
    ]


def run_all(
    data_root: Path = Path("data/part1"),
    out_dir: Path = Path("results/part1/figures"),
    traj_dir: Path = Path("results/part1/landscape_traj"),
    snapshot_every: int = 10,
    alphas: np.ndarray = np.linspace(-1.0, 1.0, 51),
    betas: np.ndarray = np.linspace(-1.0, 1.0, 51),
    adam_epochs_data: int = 100,
    adam_epochs_pinn: int = 100,
    lbfgs_max_iter: int = 100,
    batch_size_data: int = 4096,
    batch_size_pinn: int = 2048,
    early_tol: float = 1e-2,
    patience: int = 10,
    min_epochs_data: int = 50,
    min_epochs_pinn: int = 50,
    seed: int = 0,
) -> None:
    device = get_device()
    print(f"Device: {device} (cuda_available={torch.cuda.is_available()})")

    out_dir.mkdir(parents=True, exist_ok=True)
    traj_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    for cfg in best_configs():
        print(f"\n=== Landscape: method={cfg.method} K={cfg.K} sample={cfg.sample_idx} ===")

        xy, f_all, u_true, bc = load_sample(data_root, cfg.K, cfg.sample_idx, device=device)

        model = MLP(2, 1, hidden_layers=cfg.hidden_layers, width=cfg.width, seed=seed).to(device)

        if cfg.method == "datadriven":
            theta_star, snaps, final_loss = train_datadriven_with_snapshots(
                model=model,
                xy=xy,
                u_true=u_true,
                adam_lr=cfg.adam_lr,
                adam_epochs=adam_epochs_data,
                batch_size=batch_size_data,
                lbfgs_max_iter=lbfgs_max_iter,
                snapshot_every=snapshot_every,
                early_tol=early_tol,
                patience=patience,
                min_epochs=min_epochs_data,
                verbose=True,
                print_every=10,
            )
        else:
            theta_star, snaps, final_loss = train_pinn_with_snapshots(
                model=model,
                xy_all=xy,
                f_all=f_all,
                bc_mask=bc,
                lambda_bc=cfg.lambda_bc,
                adam_lr=cfg.adam_lr,
                adam_epochs=adam_epochs_pinn,
                interior_batch_size=batch_size_pinn,
                lbfgs_max_iter=lbfgs_max_iter,
                snapshot_every=snapshot_every,
                early_tol=early_tol,
                patience=patience,
                min_epochs=min_epochs_pinn,
                verbose=True,
                print_every=10,
            )

        if len(snaps) < 2:
            raise RuntimeError(f"Not enough snapshots for PCA (got {len(snaps)}). Increase adam_epochs or decrease snapshot_every.")

        d1, d2 = pca_directions(theta_star, snaps)

        tag = f"{cfg.method}_k{cfg.K}_s{cfg.sample_idx}"
        torch.save(
            {
                "cfg": cfg.__dict__,
                "theta_star": theta_star.detach().cpu(),
                "snapshots": torch.stack([s.detach().cpu() for s in snaps], dim=0),
                "d1": d1.detach().cpu(),
                "d2": d2.detach().cpu(),
                "final_loss": final_loss,
                "alphas": alphas,
                "betas": betas,
            },
            traj_dir / f"traj_{tag}.pt",
        )

        losses = eval_landscape(
            model=model,
            theta_star=theta_star,
            d1=d1,
            d2=d2,
            alphas=alphas,
            betas=betas,
            method=cfg.method,
            xy=xy,
            u_true=u_true,
            f_all=f_all,
            bc_mask=bc,
            lambda_bc=cfg.lambda_bc,
        )

        np.save(traj_dir / f"landscape_{tag}.npy", losses)

        title = f"Loss landscape (PCA) - {cfg.method.upper()}, K={cfg.K}"
        out_png = out_dir / f"landscape_{tag}.png"
        plot_contour(alphas, betas, losses, out_png=out_png, title=title)

        print(f"Saved: {out_png}")


if __name__ == "__main__":
    run_all()
