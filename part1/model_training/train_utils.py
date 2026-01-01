import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any


def rel_l2(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Relative L2 error: ||pred-target||_2 / ||target||_2.
    """
    num = torch.linalg.norm(pred - target)
    den = torch.linalg.norm(target).clamp_min(eps)
    return (num / den).item()


def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_loss_curve(losses: list[float], out_png: Path, title: str) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.5, 4), dpi=150)
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True, linestyle=":", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def save_field_grid_comparison(
    f: np.ndarray,
    u_true: np.ndarray,
    u_pred: np.ndarray,
    out_png: Path,
    title: str,
    cmap: str = "jet",
) -> None:
    """
    Save a 2x2 figure: f, u_true, u_pred, |error|.
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)

    err = np.abs(u_pred - u_true)

    fig, axs = plt.subplots(2, 2, figsize=(9, 7), dpi=150)

    im0 = axs[0, 0].imshow(f, origin="lower", cmap=cmap, aspect="auto")
    axs[0, 0].set_title("Source f")
    fig.colorbar(im0, ax=axs[0, 0], fraction=0.046, pad=0.04)

    im1 = axs[0, 1].imshow(u_true, origin="lower", cmap=cmap, aspect="auto")
    axs[0, 1].set_title("Ground Truth u")
    fig.colorbar(im1, ax=axs[0, 1], fraction=0.046, pad=0.04)

    im2 = axs[1, 0].imshow(u_pred, origin="lower", cmap=cmap, aspect="auto")
    axs[1, 0].set_title("Prediction u")
    fig.colorbar(im2, ax=axs[1, 0], fraction=0.046, pad=0.04)

    im3 = axs[1, 1].imshow(err, origin="lower", cmap=cmap, aspect="auto")
    axs[1, 1].set_title("Absolute Error |u_pred - u|")
    fig.colorbar(im3, ax=axs[1, 1], fraction=0.046, pad=0.04)

    for ax in axs.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close(fig)
