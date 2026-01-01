import itertools
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from typing import Dict, Any, Tuple
import time
from tqdm import trange

from part1.model_training.models import MLP
from part1.model_training.pde import laplacian
from part1.model_training.train_utils import rel_l2, save_json, save_loss_curve, save_field_grid_comparison


def get_device() -> torch.device:
    """
    Return CUDA device if available, else CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_example(data_root: Path, k: int, idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load (f, u) fields stored as .npy for a given K and sample index.
    """
    f = np.load(data_root / f"K{k}" / f"f_{idx}.npy")
    u = np.load(data_root / f"K{k}" / f"u_{idx}.npy")
    return f, u


def grid_points(n: int, device: torch.device) -> torch.Tensor:
    """
    Create grid coordinates (x,y) in [0,1]^2 as a tensor of shape (n*n, 2).
    """
    x = torch.linspace(0.0, 1.0, n, device=device)
    y = torch.linspace(0.0, 1.0, n, device=device)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    return torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)


def boundary_mask(n: int, device: torch.device) -> torch.Tensor:
    """
    Boolean mask for boundary nodes on an n x n grid, flattened to (n*n,).
    """
    idx = torch.arange(n * n, device=device)
    i = idx // n
    j = idx % n
    return (i == 0) | (i == n - 1) | (j == 0) | (j == n - 1)


def l2_weight_penalty(model: nn.Module) -> torch.Tensor:
    """
    Sum of squared weights (excluding biases).
    """
    reg = torch.zeros((), device=next(model.parameters()).device)
    for name, p in model.named_parameters():
        if p.requires_grad and ("weight" in name):
            reg = reg + torch.sum(p * p)
    return reg


def train_datadriven(
    model: nn.Module,
    xy: torch.Tensor,
    u_true: torch.Tensor,
    adam_lr: float,
    adam_epochs: int,
    batch_size: int,
    lbfgs_max_iter: int,
    verbose: bool,
    print_every: int,
) -> Tuple[nn.Module, list[float]]:
    """
    Train with L_data = MSE(u_pred, u_true) using Adam then L-BFGS.
    """
    ds = TensorDataset(xy, u_true)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    mse = nn.MSELoss()
    losses: list[float] = []

    opt = torch.optim.Adam(model.parameters(), lr=adam_lr)

    model.train()
    for epoch in trange(adam_epochs, disable=not verbose, desc="Adam (Data)"):
        epoch_loss = 0.0
        nb = 0
        for xb, ub in dl:
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = mse(pred, ub)
            loss.backward()
            opt.step()
            val = loss.item()
            losses.append(val)
            epoch_loss += val
            nb += 1

        mean_epoch = epoch_loss / max(nb, 1)
        if verbose and (epoch % print_every == 0 or epoch == adam_epochs - 1):
            print(f"[Data][Adam] epoch {epoch+1}/{adam_epochs}  mean_loss={mean_epoch:.6e}")

    xy_full = xy
    u_full = u_true

    lbfgs = torch.optim.LBFGS(
        model.parameters(),
        max_iter=lbfgs_max_iter,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        lbfgs.zero_grad(set_to_none=True)
        pred = model(xy_full)
        loss = mse(pred, u_full)
        loss.backward()
        return loss

    with torch.enable_grad():
        loss_before = closure().item()
        lbfgs.step(closure)
        loss_after = closure().item()
        losses.append(loss_before)
        losses.append(loss_after)

    if verbose:
        print(f"[Data][LBFGS] loss_before={loss_before:.6e}  loss_after={loss_after:.6e}")

    return model, losses


def train_pinn(
    model: nn.Module,
    xy_all: torch.Tensor,
    f_all: torch.Tensor,
    bc_mask: torch.Tensor,
    lambda_bc: float,
    adam_lr: float,
    adam_epochs: int,
    interior_batch_size: int,
    lbfgs_max_iter: int,
    verbose: bool,
    print_every: int,
) -> Tuple[nn.Module, list[float]]:
    """
    Train with L_pinn = MSE(-Δu_pred - f) on interior + lambda_bc*MSE(u_pred) on boundary.
    Collocation points are the grid nodes.
    """
    mse = nn.MSELoss()
    losses: list[float] = []

    xy_int = xy_all[~bc_mask]
    f_int = f_all[~bc_mask]
    xy_bc = xy_all[bc_mask]

    ds = TensorDataset(xy_int, f_int)
    dl = DataLoader(ds, batch_size=interior_batch_size, shuffle=True)

    opt = torch.optim.Adam(model.parameters(), lr=adam_lr)

    model.train()
    for epoch in trange(adam_epochs, disable=not verbose, desc="Adam (PINN)"):
        epoch_loss = 0.0
        nb = 0

        for xb, fb in dl:
            xb = xb.detach().requires_grad_(True)

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

            val = loss.item()
            losses.append(val)
            epoch_loss += val
            nb += 1

        mean_epoch = epoch_loss / max(nb, 1)
        if verbose and (epoch % print_every == 0 or epoch == adam_epochs - 1):
            print(f"[PINN][Adam] epoch {epoch+1}/{adam_epochs}  mean_loss={mean_epoch:.6e}")

    xy_int_full = xy_int.detach().requires_grad_(True)
    f_int_full = f_int
    xy_bc_full = xy_bc

    lbfgs = torch.optim.LBFGS(
        model.parameters(),
        max_iter=lbfgs_max_iter,
        line_search_fn="strong_wolfe",
    )

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
        loss_before = closure().item()
        lbfgs.step(closure)
        loss_after = closure().item()
        losses.append(loss_before)
        losses.append(loss_after)

    if verbose:
        print(f"[PINN][LBFGS] loss_before={loss_before:.6e}  loss_after={loss_after:.6e}")

    return model, losses


def evaluate_on_grid(
    model: nn.Module,
    xy: torch.Tensor,
    u_true: torch.Tensor,
    n: int,
) -> Tuple[float, np.ndarray]:
    """
    Evaluate relative L2 error and return u_pred reshaped to (n,n).
    """
    model.eval()
    with torch.no_grad():
        pred = model(xy)
        err = rel_l2(pred, u_true)
        u_pred = pred.detach().cpu().numpy().reshape(n, n)
    return err, u_pred


def grid_search_configs() -> list[Dict[str, Any]]:
    """
    Medium grid search over architecture and optimization hyperparameters.
    """
    hidden_layers = [3, 4]
    widths = [32, 64, 128]
    adam_lrs = [1e-3, 5e-4]
    # reg_lambdas = [0.0, 1e-7, 1e-6]
    lambda_bcs = [10.0, 50.0, 100.0]

    configs = []
    for hl, w, lr, lbc in itertools.product(hidden_layers, widths, adam_lrs, lambda_bcs):
        configs.append(
            {
                "hidden_layers": hl,
                "width": w,
                "adam_lr": lr,
                "lambda_bc": lbc,
            }
        )
    return configs


def run_one(
    method: str,
    k: int,
    f_np: np.ndarray,
    u_np: np.ndarray,
    device: torch.device,
    out_weights: Path,
    out_figs: Path,
    sample_idx: int,
    seed: int,
) -> Dict[str, Any]:
    """
    Run grid search for a given K and method (datadriven or pinn), save best model, and export plots.
    """
    print(f"\n=== Grid search start with method={method} ; K={k} ; sample={sample_idx} ===")
    t0_global = time.time()

    n = f_np.shape[0]
    xy = grid_points(n, device)
    bc = boundary_mask(n, device)

    f = torch.tensor(f_np.reshape(-1, 1), dtype=torch.float32, device=device)
    u = torch.tensor(u_np.reshape(-1, 1), dtype=torch.float32, device=device)

    configs = grid_search_configs()
    results: list[Dict[str, Any]] = []

    best = {"rel_l2": float("inf")}
    best_state = None
    best_losses: list[float] = []
    best_u_pred = None
    best_cfg = None

    for cfg_id, cfg in enumerate(configs):
        model = MLP(2, 1, hidden_layers=cfg["hidden_layers"], width=cfg["width"], seed=seed).to(device)

        t0 = time.time()
        print(
            f"[{method.upper()}][K={k}] cfg {cfg_id+1}/{len(configs)} "
            f"hidden_layers={cfg['hidden_layers']} width={cfg['width']} adam_lr={cfg['adam_lr']} "
            + (f"lambda_bc={cfg['lambda_bc']}" if method == "pinn" else "")
        )

        if method == "datadriven":
            model, losses = train_datadriven(
                model=model,
                xy=xy,
                u_true=u,
                # reg_lambda=cfg["reg_lambda"],
                adam_lr=cfg["adam_lr"],
                adam_epochs=5000,
                batch_size=4096,
                lbfgs_max_iter=500,
                verbose=True,
                print_every=200,
            )
        elif method == "pinn":
            model, losses = train_pinn(
                model=model,
                xy_all=xy,
                f_all=f,
                bc_mask=bc,
                reg_lambda=cfg["reg_lambda"],
                lambda_bc=cfg["lambda_bc"],
                adam_lr=cfg["adam_lr"],
                adam_epochs=5000,
                interior_batch_size=2048,
                lbfgs_max_iter=500,
                verbose=True,
                print_every=200,
            )
        else:
            raise ValueError("method must be 'datadriven' or 'pinn'")

        err, u_pred = evaluate_on_grid(model, xy, u, n)

        print(
            f"[{method.upper()}][K={k}] cfg {cfg_id+1}/{len(configs)} done "
            f"rel_l2={err:.6e} final_loss={losses[-1]:.6e} elapsed={time.time()-t0:.1f}s "
            f"best_so_far={best['rel_l2']:.6e}"
        )

        row = {
            "cfg_id": cfg_id,
            "method": method,
            "K": k,
            "sample_idx": sample_idx,
            "hidden_layers": cfg["hidden_layers"],
            "width": cfg["width"],
            "adam_lr": cfg["adam_lr"],
            "reg_lambda": cfg["reg_lambda"],
            "lambda_bc": cfg["lambda_bc"],
            "rel_l2": err,
            "final_loss": float(losses[-1]) if len(losses) > 0 else None,
        }
        results.append(row)

        if err < best["rel_l2"]:
            best = row
            best_state = {kk: vv.detach().cpu() for kk, vv in model.state_dict().items()}
            best_losses = losses
            best_u_pred = u_pred
            best_cfg = cfg

    out_weights.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    tag = f"{method}_k{k}_s{sample_idx}"
    weights_path = out_weights / f"mlp_{tag}_best.pt"
    torch.save(
        {
            "best": best,
            "config": best_cfg,
            "state_dict": best_state,
        },
        weights_path,
    )

    loss_png = out_figs / f"loss_curve_{tag}.png"
    save_loss_curve(best_losses, loss_png, title=f"Training Loss ({method.upper()}, K={k})")

    pred_png = out_figs / f"fields_{tag}.png"
    save_field_grid_comparison(
        f=f_np,
        u_true=u_np,
        u_pred=best_u_pred,
        out_png=pred_png,
        title=f"Fields (method={method.upper()}, K={k})",
        cmap="jet",
    )

    summary_path = out_figs / f"grid_search_{tag}.json"
    save_json({"best": best, "all": results}, summary_path)


    print(f"=== Grid search done | method={method} | K={k} ===")
    print(f"Best rel_l2: {best['rel_l2']:.6e}")
    print(f"Best cfg: hidden_layers={best['hidden_layers']} width={best['width']} adam_lr={best['adam_lr']} "
        + (f"lambda_bc={best['lambda_bc']}" if method == "pinn" else ""))
    print(f"Saved weights: {weights_path}")
    print(f"Saved plots: {loss_png} and {pred_png}")
    print(f"Total time: {time.time()-t0_global:.1f}s")

    return {"best": best, "weights_path": str(weights_path), "loss_png": str(loss_png), "pred_png": str(pred_png)}


def main() -> None:
    device = get_device()
    print(f"Device: {device} (cuda_available={torch.cuda.is_available()})")

    data_root = Path("data/part1")
    out_weights = Path("part1/trained_weights")
    out_figs = Path("results/part1/figures")

    ks = [1, 4, 16]
    sample_idx = 0
    seed = 0

    all_runs = []

    for k in ks:
        f_np, u_np = load_example(data_root, k, sample_idx)
        for method in ["datadriven", "pinn"]:
            run_info = run_one(
                method=method,
                k=k,
                f_np=f_np,
                u_np=u_np,
                device=device,
                out_weights=out_weights,
                out_figs=out_figs,
                sample_idx=sample_idx,
                seed=seed,
            )
            all_runs.append(run_info)

    save_json({"runs": all_runs, "device": str(device)}, out_figs / "task12_overview.json")


if __name__ == "__main__":
    main()
