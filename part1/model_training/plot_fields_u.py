import re
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

from part1.model_training.models import MLP


_FILENAME_RE = re.compile(r"^mlp_(datadriven|pinn)_k(\d+)_s(\d+)_best\.pt$")


def get_device() -> torch.device:
    """
    Return CUDA device if available, else CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def grid_points(n: int, device: torch.device) -> torch.Tensor:
    """
    Create grid coordinates (x,y) in [0,1]^2 as a tensor of shape (n*n, 2).
    """
    x = torch.linspace(0.0, 1.0, n, device=device)
    y = torch.linspace(0.0, 1.0, n, device=device)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    return torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)


def rel_l2(pred: np.ndarray, target: np.ndarray, eps: float = 1e-12) -> float:
    """
    Relative L2 error: ||pred-target||_2 / ||target||_2.
    """
    num = np.linalg.norm(pred - target)
    den = max(np.linalg.norm(target), eps)
    return float(num / den)


def save_u_fields_png(
    u_true: np.ndarray,
    u_pred: np.ndarray,
    out_png: Path,
    title: str,
    cmap: str = "jet",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """
    Save a 1x3 figure: Ground Truth u, Prediction u, Absolute Error |u_pred - u|.
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)

    err = np.abs(u_pred - u_true)

    if vmin is None:
        vmin = float(min(u_true.min(), u_pred.min()))
    if vmax is None:
        vmax = float(max(u_true.max(), u_pred.max()))

    fig, axs = plt.subplots(1, 3, figsize=(12, 4), dpi=150)

    im0 = axs[0].imshow(u_true, origin="lower", cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    axs[0].set_title("Ground Truth u")
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].imshow(u_pred, origin="lower", cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    axs[1].set_title("Prediction u")
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    im2 = axs[2].imshow(err, origin="lower", cmap=cmap, aspect="auto")
    axs[2].set_title("Absolute Error |u_pred - u|")
    fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def load_checkpoint(path: Path, device: torch.device) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Load a checkpoint saved by torch.save({best, config, state_dict}, ...).
    """
    ckpt = torch.load(path, map_location=device)
    config = ckpt.get("config", {})
    state_dict = ckpt.get("state_dict", {})
    best = ckpt.get("best", {})
    return config, state_dict, best


def predict_on_grid(
    model: torch.nn.Module,
    xy: torch.Tensor,
    n: int,
    batch_size: int = 65536,
) -> np.ndarray:
    """
    Predict u on the full grid, returning an (n,n) numpy array.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for i in range(0, xy.shape[0], batch_size):
            preds.append(model(xy[i : i + batch_size]).detach().cpu().numpy())

    pred = np.concatenate(preds, axis=0).reshape(n, n)
    return pred


def render_all_fields(
    weights_dir: Path = Path("part1/trained_weights"),
    data_root: Path = Path("data/part1"),
    out_figs: Path = Path("results/part1/figures"),
) -> None:
    """
    Render U-only field PNGs for all saved .pt weights found in weights_dir.
    """
    device = get_device()
    print(f"Device: {device} (cuda_available={torch.cuda.is_available()})")

    ckpts = sorted(weights_dir.glob("*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No .pt files found in {weights_dir}")

    rendered = 0
    skipped = 0

    for p in ckpts:
        m = _FILENAME_RE.match(p.name)
        if m is None:
            skipped += 1
            continue

        method = m.group(1)
        k = int(m.group(2))
        s = int(m.group(3))

        u_path = data_root / f"K{k}" / f"u_{s}.npy"
        if not u_path.exists():
            print(f"Skipping {p.name}: missing {u_path}")
            skipped += 1
            continue

        u_true = np.load(u_path)
        n = int(u_true.shape[0])

        config, state_dict, best = load_checkpoint(p, device)

        hidden_layers = int(config.get("hidden_layers"))
        width = int(config.get("width"))

        model = MLP(2, 1, hidden_layers=hidden_layers, width=width, seed=0).to(device)
        model.load_state_dict(state_dict, strict=True)

        xy = grid_points(n, device)
        u_pred = predict_on_grid(model, xy, n=n)

        err = rel_l2(u_pred, u_true)
        err_saved = best.get("rel_l2", None)
        err_str = f"{err:.3e}" if err is not None else "NA"
        err_saved_str = f"{float(err_saved):.3e}" if err_saved is not None else "NA"

        title = (
            f"U Fields ({method.upper()}, K={k}, sample={s})  "
            f"rel_L2={err_str}"
        )

        out_png = out_figs / f"u_fields_{method}_k{k}_s{s}.png"
        save_u_fields_png(u_true=u_true, u_pred=u_pred, out_png=out_png, title=title, cmap="jet")

        print(f"Rendered: {out_png}")
        rendered += 1

    print(f"Done. Rendered={rendered}, Skipped={skipped}")


if __name__ == "__main__":
    render_all_fields()
