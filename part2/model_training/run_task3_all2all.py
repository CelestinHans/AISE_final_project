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
    err_n = ||pred-true||_2 / ||true||_2  (per sample)
    pred,true: (B,S) or (B,S,1)
    returns: (B,)
    """
    if pred.ndim == 3:
        pred = pred.squeeze(-1)
    if true.ndim == 3:
        true = true.squeeze(-1)
    num = torch.linalg.vector_norm(pred - true, ord=2, dim=1)
    den = torch.linalg.vector_norm(true, ord=2, dim=1).clamp_min(eps)
    return num / den


def save_curve(train_vals, val_vals, out_path: Path, title: str, ylabel: str, yscale: str = "linear"):
    epochs = np.arange(1, len(train_vals) + 1)
    plt.figure()
    plt.plot(epochs, train_vals, label="train")
    plt.plot(epochs, val_vals, label="val")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.yscale(yscale)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()



class All2AllTrajectoryDataset(Dataset):
    """
    Loads .npy of shape (N, 5, S) with snapshots at times:
      [0.0, 0.25, 0.50, 0.75, 1.0]
    Creates training pairs (i,j) with i<j:
      u(t_j) = S(t_j - t_i, u(t_i))
    Sample:
      input  (S,3) = [u(t_i)(x), x, Δt]
      target (S,1) = u(t_j)(x)
    Total samples = N * len(time_pairs) where len(time_pairs)=10 for T=5.
    """
    def __init__(self, npy_path: Path, s_expected: int = 128):
        arr = np.load(npy_path)  # (N,5,S)
        assert arr.ndim == 3 and arr.shape[1] == 5, f"Bad shape {arr.shape}, expected (N,5,S)"
        self.N, self.T, self.S = arr.shape
        assert self.S == s_expected, f"Expected S={s_expected}, got S={self.S} in {npy_path.name}"

        self.u = torch.from_numpy(arr).float()  # (N,T,S)

        self.times = torch.tensor([0.0, 0.25, 0.50, 0.75, 1.0]).float()  # (T,)
        self.time_pairs = [(i, j) for i in range(0, self.T) for j in range(i + 1, self.T)]
        self.len_times = len(self.time_pairs)  # 10

        self.x = torch.linspace(0.0, 1.0, self.S).float()  # (S,)

    def __len__(self) -> int:
        return self.N * self.len_times

    def __getitem__(self, idx: int):
        n = idx // self.len_times
        p = idx % self.len_times
        i, j = self.time_pairs[p]

        ui = self.u[n, i, :]  # (S,)
        uj = self.u[n, j, :]  # (S,)
        dt = self.times[j] - self.times[i]  # scalar

        dt_chan = torch.full_like(self.x, dt)              # (S,)
        inp = torch.stack([ui, self.x, dt_chan], dim=-1)   # (S,3)
        tgt = uj[:, None]                                  # (S,1)
        return inp, tgt


#time-conditional normalization
class FILM(nn.Module):
    def __init__(self, channels, use_bn: bool = True):
        super().__init__()
        self.channels = channels

        self.inp2scale = nn.Linear(in_features=1, out_features=channels, bias=True)
        self.inp2bias = nn.Linear(in_features=1, out_features=channels, bias=True)

        self.inp2scale.weight.data.fill_(1)
        self.inp2scale.bias.data.fill_(0)
        self.inp2bias.weight.data.fill_(0)
        self.inp2bias.bias.data.fill_(0)

        if use_bn:
            self.norm = nn.BatchNorm1d(channels)
        else:
            self.norm = nn.Identity()

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        # x: (B,C,S), time: (B,) or (B,1)
        x = self.norm(x)

        time = time.reshape(-1, 1).type_as(x)  # (B,1)
        scale = self.inp2scale(time)           # (B,C)
        bias = self.inp2bias(time)             # (B,C)

        scale = scale.unsqueeze(2).expand_as(x)  # (B,C,S)
        bias = bias.unsqueeze(2).expand_as(x)    # (B,C,S)

        return x * (1.0 + scale) + bias



#Spectral layer (same as task 2)
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
        # x: (B,in_ch,S)
        x_ft = torch.fft.rfft(x)  # (B,in_ch,S//2+1)
        modes = min(self.modes1, x_ft.size(-1))

        out_ft = torch.zeros(
            (x.size(0), self.out_channels, x_ft.size(-1)),
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, :modes] = self.compl_mul1d(x_ft[:, :, :modes], self.weights1[:, :, :modes])
        x = torch.fft.irfft(out_ft, n=x.shape[-1], dim=-1)  # (B,out_ch,S)
        return x


class TimeDependentFNO1d_All2All(nn.Module):
    """
    Input:  (B,S,3) = [u(t_i)(x), x, Δt=t_j-t_i]
    Output: (B,S,1) = u(t_j)(x)

    Uses FiLM + BatchNorm1d conditioned on Δt (time-conditional normalization).
    """
    def __init__(self, modes: int = 16, width: int = 64, use_bn: bool = True):
        super().__init__()
        self.modes = modes
        self.width = width

        self.linear_p = nn.Linear(3, width)

        self.spect1 = SpectralConv1d(width, width, modes)
        self.spect2 = SpectralConv1d(width, width, modes)
        self.spect3 = SpectralConv1d(width, width, modes)
        self.spect4 = SpectralConv1d(width, width, modes)

        self.lin0 = nn.Conv1d(width, width, 1)
        self.lin1 = nn.Conv1d(width, width, 1)
        self.lin2 = nn.Conv1d(width, width, 1)
        self.lin3 = nn.Conv1d(width, width, 1)

        self.film1 = FILM(width, use_bn=use_bn)
        self.film2 = FILM(width, use_bn=use_bn)
        self.film3 = FILM(width, use_bn=use_bn)
        self.film4 = FILM(width, use_bn=use_bn)

        self.act = nn.Tanh()

        self.linear_q = nn.Linear(width, 64)
        self.output_layer = nn.Linear(64, 1)

    def _block(self, x: torch.Tensor, dt: torch.Tensor, spectral: nn.Module, pointwise: nn.Module, film: nn.Module) -> torch.Tensor:
        # x: (B,C,S), dt: (B,)
        x = spectral(x) + pointwise(x)
        x = film(x, dt)
        x = self.act(x)
        return x

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        # inp: (B,S,3). Δt is constant along S -> read from first spatial point
        dt = inp[:, 0, 2]  # (B,)

        x = self.linear_p(inp)   # (B,S,C)
        x = x.permute(0, 2, 1)   # (B,C,S)

        x = self._block(x, dt, self.spect1, self.lin0, self.film1)
        x = self._block(x, dt, self.spect2, self.lin1, self.film2)
        x = self._block(x, dt, self.spect3, self.lin2, self.film3)
        x = self._block(x, dt, self.spect4, self.lin3, self.film4)

        x = x.permute(0, 2, 1)   # (B,S,C)
        x = self.act(self.linear_q(x))
        x = self.output_layer(x) # (B,S,1)
        return x


@dataclass
class Task3Config:
    seed: int = 0
    batch_size: int = 32
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-4
    modes: int = 16
    width: int = 64
    use_bn: bool = True
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def run_epoch(model, loader, optimizer=None, device="cpu"):
    train = optimizer is not None
    model.train(train)

    mse = nn.MSELoss()
    loss_sum = 0.0
    rel_sum = 0.0
    n_samples = 0

    for inp, tgt in loader:
        inp = inp.to(device)  # (B,S,3)
        tgt = tgt.to(device)  # (B,S,1)

        pred = model(inp)
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


def main():
    cfg = Task3Config()
    set_seed(cfg.seed)

    data_dir = Path("data/part2/FNO_data")
    results_dir = Path("results/part2/task3_all2all_time_dependent")
    results_dir.mkdir(parents=True, exist_ok=True)


    results_part2_dir = Path("results/part2")
    results_part2_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "data_train_128.npy"
    val_path = data_dir / "data_val_128.npy"

    train_ds = All2AllTrajectoryDataset(train_path, s_expected=128)
    val_ds = All2AllTrajectoryDataset(val_path, s_expected=128)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=(cfg.device == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=(cfg.device == "cuda")
    )

    model = TimeDependentFNO1d_All2All(modes=cfg.modes, width=cfg.width, use_bn=cfg.use_bn).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    best_val = float("inf")
    best_path = results_dir / "best_model.pt"

    train_mse_hist, val_mse_hist = [], []
    train_rel_hist, val_rel_hist = [], []

    log_path = results_dir / "log.txt"
    with open(log_path, "w") as f:
        f.write("epoch,train_mse,train_relL2,val_mse,val_relL2,lr\n")

    for epoch in range(1, cfg.epochs + 1):
        train_mse, train_rel = run_epoch(model, train_loader, optimizer=optimizer, device=cfg.device)
        val_mse, val_rel = run_epoch(model, val_loader, optimizer=None, device=cfg.device)
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


    save_curve(
        train_mse_hist, val_mse_hist,
        out_path=results_part2_dir / "loss_curve_task3.png",
        title="Task 3: Training/Validation MSE loss",
        ylabel="MSE",
        yscale="linear",
    )

    save_curve(
        train_mse_hist, val_mse_hist,
        out_path=results_part2_dir / "loss_curve_task3_log.png",
        title="Task 3: Training/Validation MSE loss (log scale)",
        ylabel="MSE",
        yscale="log",
    )
    save_curve(
        train_rel_hist, val_rel_hist,
        out_path=results_part2_dir / "relL2_curve_task3.png",
        title="Task 3: Training/Validation relative L2 (statement metric)",
        ylabel="Relative L2",
        yscale="linear",
    )

    print(f"\nSaved best checkpoint to: {best_path}")
    print(f"Saved plots to: {results_part2_dir}")


if __name__ == "__main__":
    main()
