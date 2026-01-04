import os
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt


def set_seed(seed: int = 0) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def relative_l2(pred: torch.Tensor, true: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    err_n = ||pred-true||_2 / ||true||_2   (per trajectory)
    pred, true: (B, S) or (B, S, 1)
    returns: (B,)
    """
    if pred.ndim == 3:
        pred = pred.squeeze(-1)
    if true.ndim == 3:
        true = true.squeeze(-1)
    num = torch.linalg.vector_norm(pred - true, ord=2, dim=1)
    den = torch.linalg.vector_norm(true, ord=2, dim=1).clamp_min(eps)
    return num / den


class One2OneTrajectoryDataset(Dataset):
    def __init__(self, npy_path: Path, s_expected: int = 128):
        arr = np.load(npy_path)  # (N, 5, S)
        assert arr.ndim == 3 and arr.shape[1] == 5, f"Bad shape {arr.shape}, expected (N,5,S)"
        self.N, _, self.S = arr.shape
        assert self.S == s_expected, f"Expected S={s_expected}, got S={self.S} in {npy_path.name}"

        self.u0 = torch.from_numpy(arr[:, 0, :]).float()  # (N, S)
        self.uT = torch.from_numpy(arr[:, 4, :]).float()  # (N, S)

        x = torch.linspace(0.0, 1.0, self.S).float()      # (S,)
        self.x = x[None, :].repeat(self.N, 1)             # (N, S)

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int):
        u0 = self.u0[idx]                  # (S,)
        x = self.x[idx]                    # (S,)
        inp = torch.stack([u0, x], dim=-1) # (S,2)
        tgt = self.uT[idx][:, None]        # (S,1)
        return inp, tgt


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, dtype=torch.cfloat)
        )

    def compl_mul1d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ft = torch.fft.rfft(x)  # (B, in_ch, S//2+1)
        modes = min(self.modes1, x_ft.size(-1))

        out_ft = torch.zeros(
            (x.size(0), self.out_channels, x_ft.size(-1)),
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, :modes] = self.compl_mul1d(x_ft[:, :, :modes], self.weights1[:, :, :modes])
        x = torch.fft.irfft(out_ft, n=x.shape[-1], dim=-1)
        return x


class FNO1d(nn.Module):
    def __init__(self, modes: int = 16, width: int = 64):
        super().__init__()
        self.linear_p = nn.Linear(2, width)

        self.spect1 = SpectralConv1d(width, width, modes)
        self.spect2 = SpectralConv1d(width, width, modes)
        self.spect3 = SpectralConv1d(width, width, modes)
        self.spect4 = SpectralConv1d(width, width, modes)

        self.lin0 = nn.Conv1d(width, width, 1)
        self.lin1 = nn.Conv1d(width, width, 1)
        self.lin2 = nn.Conv1d(width, width, 1)
        self.lin3 = nn.Conv1d(width, width, 1)

        self.linear_q = nn.Linear(width, 64)
        self.output_layer = nn.Linear(64, 1)

        self.act = nn.Tanh()

    def _fno_block(self, x: torch.Tensor, spectral: nn.Module, pointwise: nn.Module) -> torch.Tensor:
        return self.act(spectral(x) + pointwise(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_p(x)          # (B,S,C)
        x = x.permute(0, 2, 1)        # (B,C,S)

        x = self._fno_block(x, self.spect1, self.lin0)
        x = self._fno_block(x, self.spect2, self.lin1)
        x = self._fno_block(x, self.spect3, self.lin2)
        x = self._fno_block(x, self.spect4, self.lin3)

        x = x.permute(0, 2, 1)        # (B,S,C)
        x = self.act(self.linear_q(x))
        x = self.output_layer(x)      # (B,S,1)
        return x


@dataclass
class Task1Config:
    seed: int = 0
    batch_size: int = 32
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-4
    modes: int = 16
    width: int = 64
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    plot_test_index: int = 0


def run_epoch(model, loader, optimizer=None, device="cpu"):
    train = optimizer is not None
    model.train(train)

    mse = nn.MSELoss()
    loss_sum = 0.0
    rel_sum = 0.0
    n_samples = 0

    for inp, tgt in loader:
        inp = inp.to(device)  # (B,S,2)
        tgt = tgt.to(device)  # (B,S,1)

        pred = model(inp)     # (B,S,1)
        loss = mse(pred, tgt)

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        B = inp.size(0)
        loss_sum += loss.item() * B
        rel_sum += relative_l2(pred, tgt).sum().item()
        n_samples += B

    return loss_sum / n_samples, rel_sum / n_samples


@torch.no_grad()
def predict_all(model, loader, device="cpu"):
    model.eval()
    preds_list, trues_list = [], []
    for inp, tgt in loader:
        inp = inp.to(device)
        tgt = tgt.to(device)
        pred = model(inp)  # (B,S,1)
        preds_list.append(pred.squeeze(-1).detach().cpu())
        trues_list.append(tgt.squeeze(-1).detach().cpu())
    return torch.cat(preds_list, dim=0), torch.cat(trues_list, dim=0)


def save_loss_curve(train_vals, val_vals, out_path: Path, title: str, yscale: str = "linear"):
    epochs = np.arange(1, len(train_vals) + 1)
    plt.figure()
    plt.plot(epochs, train_vals, label="train")
    plt.plot(epochs, val_vals, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title(title)
    plt.yscale(yscale)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_single_trajectory_plot(x, pred, true, out_path: Path):
    plt.figure()
    plt.plot(x, true, label="true u(t=1)")
    plt.plot(x, pred, label="pred u(t=1)")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.title("Test trajectory: prediction vs true at t=1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_mean_trajectory_plot(x, pred_mean, true_mean, out_path: Path):
    plt.figure()
    plt.plot(x, true_mean, label="mean true u(t=1)")
    plt.plot(x, pred_mean, label="mean pred u(t=1)")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.title("Mean over test trajectories at t=1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_test_relL2_barplot(rel_errors: np.ndarray, out_path: Path):
    plt.figure(figsize=(10, 4))
    plt.bar(np.arange(len(rel_errors)), rel_errors)
    plt.xlabel("Test trajectory index")
    plt.ylabel("Relative L2 error")
    plt.title("Relative L2 error per test trajectory (t=1)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    cfg = Task1Config()
    set_seed(cfg.seed)

    data_dir = Path("data/part2/FNO_data")
    results_dir = Path("results/part2/task1_one2one")
    results_dir.mkdir(parents=True, exist_ok=True)

    # As requested: all plots into results/part2
    results_part2_dir = Path("results/part2")
    results_part2_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "data_train_128.npy"
    val_path   = data_dir / "data_val_128.npy"
    test_path  = data_dir / "data_test_128.npy"

    train_ds = One2OneTrajectoryDataset(train_path, s_expected=128)
    val_ds   = One2OneTrajectoryDataset(val_path, s_expected=128)
    test_ds  = One2OneTrajectoryDataset(test_path, s_expected=128)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=(cfg.device == "cuda"))
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=(cfg.device == "cuda"))
    test_loader  = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=(cfg.device == "cuda"))

    model = FNO1d(modes=cfg.modes, width=cfg.width).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    best_val = float("inf")
    best_path = results_dir / "best_model.pt"

    log_path = results_dir / "log.txt"
    with open(log_path, "w") as f:
        f.write("epoch,train_mse,train_relL2,val_mse,val_relL2,lr\n")

    train_mse_hist, val_mse_hist = [], []
    train_rel_hist, val_rel_hist = [], []

    for epoch in range(1, cfg.epochs + 1):
        train_mse, train_rel = run_epoch(model, train_loader, optimizer=optimizer, device=cfg.device)
        val_mse, val_rel     = run_epoch(model, val_loader, optimizer=None, device=cfg.device)
        scheduler.step(val_mse)

        train_mse_hist.append(train_mse)
        val_mse_hist.append(val_mse)
        train_rel_hist.append(train_rel)
        val_rel_hist.append(val_rel)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"[{epoch:03d}/{cfg.epochs}] "
              f"train: mse={train_mse:.4e}, relL2={train_rel:.4e} | "
              f"val: mse={val_mse:.4e}, relL2={val_rel:.4e} | lr={lr_now:.2e}")

        with open(log_path, "a") as f:
            f.write(f"{epoch},{train_mse},{train_rel},{val_mse},{val_rel},{lr_now}\n")

        if val_rel < best_val:
            best_val = val_rel
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, best_path)

 
    save_loss_curve(
        train_mse_hist, val_mse_hist,
        out_path=results_part2_dir / "loss_curve_task1.png",
        title="Training / Validation MSE loss",
        yscale="linear",
    )
    save_loss_curve(
        train_mse_hist, val_mse_hist,
        out_path=results_part2_dir / "loss_curve_task1_log.png",
        title="Training / Validation MSE loss (log scale)",
        yscale="log",
    )


    save_loss_curve(
        train_rel_hist, val_rel_hist,
        out_path=results_part2_dir / "relL2_curve_task1.png",
        title="Training / Validation relative L2 (statement metric)",
        yscale="linear",
    )


    ckpt = torch.load(best_path, map_location=cfg.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    _, test_rel = run_epoch(model, test_loader, optimizer=None, device=cfg.device)
    print("\n===== TASK 1 RESULT =====")
    print(f"Average relative L2 error on data_test_128.npy (N=128): {test_rel:.6f}")

    with open(results_dir / "test_result.txt", "w") as f:
        f.write(f"Average relative L2 error (test_128): {test_rel:.6f}\n")

    preds, trues = predict_all(model, test_loader, device=cfg.device)  # (N,S) cpu
    S = preds.shape[1]
    x = np.linspace(0.0, 1.0, S)


    idx = int(np.clip(cfg.plot_test_index, 0, preds.shape[0] - 1))
    save_single_trajectory_plot(
        x=x,
        pred=preds[idx].numpy(),
        true=trues[idx].numpy(),
        out_path=results_part2_dir / "test_single_trajectory_t1.png",
    )

    save_mean_trajectory_plot(
        x=x,
        pred_mean=preds.mean(dim=0).numpy(),
        true_mean=trues.mean(dim=0).numpy(),
        out_path=results_part2_dir / "test_mean_trajectory_t1.png",
    )


    rel_per_traj = relative_l2(preds, trues).numpy()  # (N,)
    save_test_relL2_barplot(
        rel_errors=rel_per_traj,
        out_path=results_part2_dir / "test_relL2_barplot.png",
    )


if __name__ == "__main__":
    main()
