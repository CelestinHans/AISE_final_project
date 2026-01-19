
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
    err_n = ||pred-true||_2 / ||true||_2
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
    .npy shape: (N, 5, S)
    snapshots at times: [0.0, 0.25, 0.50, 0.75, 1.0]
    all2all pairs (i<j):
      input  (S,3) = [u(t_i)(x), x, dt=t_j-t_i]
      target (S,1) = u(t_j)(x)
    """
    def __init__(self, npy_path: Path, s_expected: int = 128):
        arr = np.load(npy_path)  # (N,5,S)
        assert arr.ndim == 3 and arr.shape[1] == 5, f"Bad shape {arr.shape}, expected (N,5,S)"
        self.N, self.T, self.S = arr.shape
        assert self.S == s_expected, f"Expected S={s_expected}, got S={self.S} in {npy_path.name}"

        self.u = torch.from_numpy(arr).float()  # (N,T,S)

        self.times = torch.tensor([0.0, 0.25, 0.50, 0.75, 1.0]).float()
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
        dt = self.times[j] - self.times[i]

        dt_chan = torch.full_like(self.x, dt)
        inp = torch.stack([ui, self.x, dt_chan], dim=-1)  # (S,3)
        tgt = uj[:, None]                                 # (S,1)
        return inp, tgt



class FILM(nn.Module):
    def __init__(self, channels, use_bn: bool = True):
        super(FILM, self).__init__()
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

    def forward(self, x, time):
        x = self.norm(x)
        time = time.reshape(-1, 1).type_as(x)
        scale = self.inp2scale(time)
        bias = self.inp2bias(time)
        scale = scale.unsqueeze(2).expand_as(x)
        bias = bias.unsqueeze(2).expand_as(x)
        return x * (1.0 + scale) + bias



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
        x_ft = torch.fft.rfft(x)
        modes = min(self.modes1, x_ft.size(-1))
        out_ft = torch.zeros(
            (x.size(0), self.out_channels, x_ft.size(-1)),
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, :modes] = self.compl_mul1d(x_ft[:, :, :modes], self.weights1[:, :, :modes])
        x = torch.fft.irfft(out_ft, n=x.shape[-1], dim=-1)
        return x


class TimeDependentFNO1d_All2All(nn.Module):
    """
    Input:  (B,S,3) = [u(t_i)(x), x, dt]
    Output: (B,S,1) = u(t_j)(x)
    """
    def __init__(self, modes: int = 16, width: int = 64, use_bn: bool = True):
        super().__init__()
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
        x = spectral(x) + pointwise(x)
        x = film(x, dt)
        x = self.act(x)
        return x

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        dt = inp[:, 0, 2]  # (B,)
        x = self.linear_p(inp)     # (B,S,C)
        x = x.permute(0, 2, 1)     # (B,C,S)

        x = self._block(x, dt, self.spect1, self.lin0, self.film1)
        x = self._block(x, dt, self.spect2, self.lin1, self.film2)
        x = self._block(x, dt, self.spect3, self.lin2, self.film3)
        x = self._block(x, dt, self.spect4, self.lin3, self.film4)

        x = x.permute(0, 2, 1)     # (B,S,C)
        x = self.act(self.linear_q(x))
        x = self.output_layer(x)   # (B,S,1)
        return x



@dataclass
class ScratchConfig:
    seed: int = 0
    batch_size: int = 64
    epochs: int = 150
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
        inp = inp.to(device)
        tgt = tgt.to(device)

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


@torch.no_grad()
def eval_unknown_test_t1(model: nn.Module, test_npy_path: Path, batch_size: int, device: str):
    """
    Evaluate on data_test_unknown_128.npy at t=1.0:
      input  = [u(0)(x), x, dt=1.0]
      target = u(1)(x)
    Returns avg relative L2.
    """
    arr = np.load(test_npy_path)  # (N,5,128)
    assert arr.ndim == 3 and arr.shape[1] == 5, f"Bad shape {arr.shape}, expected (N,5,S)"
    N, _, S = arr.shape

    u0 = torch.from_numpy(arr[:, 0, :]).float()  # (N,S)
    u1 = torch.from_numpy(arr[:, 4, :]).float()  # (N,S)

    x = torch.linspace(0.0, 1.0, S).float()[None, :].repeat(N, 1)  # (N,S)
    dt = torch.full((N, S), 1.0).float()                           # (N,S)

    inp = torch.stack([u0, x, dt], dim=-1)      # (N,S,3)
    tgt = u1[:, :, None]                        # (N,S,1)

    model.eval()
    rel_sum = 0.0
    n_done = 0

    for i in range(0, N, batch_size):
        inp_b = inp[i:i+batch_size].to(device)
        tgt_b = tgt[i:i+batch_size].to(device)
        pred_b = model(inp_b)
        rel_sum += relative_l2(pred_b, tgt_b).sum().item()
        n_done += inp_b.size(0)

    return rel_sum / n_done


def main():
    cfg = ScratchConfig()
    set_seed(cfg.seed)

    data_dir = Path("data/part2/FNO_data")
    train_path = data_dir / "data_finetune_train_unknown_128.npy"
    val_path   = data_dir / "data_finetune_val_unknown_128.npy"
    test_path  = data_dir / "data_test_unknown_128.npy"

    results_dir = Path("results/part2/task4_from_scratch_all2all_unknown")
    results_dir.mkdir(parents=True, exist_ok=True)

    results_part2_dir = Path("results/part2")
    results_part2_dir.mkdir(parents=True, exist_ok=True)

    train_ds = All2AllTrajectoryDataset(train_path, s_expected=128)
    val_ds   = All2AllTrajectoryDataset(val_path, s_expected=128)

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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=15)

    best_val = float("inf")
    best_path = results_dir / "best_model.pt"

    train_rel_hist, val_rel_hist = [], []

    log_path = results_dir / "log.txt"
    with open(log_path, "w") as f:
        f.write("epoch,train_relL2,val_relL2,lr\n")

    for epoch in range(1, cfg.epochs + 1):
        _, train_rel = run_epoch(model, train_loader, optimizer=optimizer, device=cfg.device)
        _, val_rel   = run_epoch(model, val_loader, optimizer=None, device=cfg.device)
        scheduler.step(val_rel)

        train_rel_hist.append(train_rel)
        val_rel_hist.append(val_rel)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"[{epoch:03d}/{cfg.epochs}] train relL2={train_rel:.4e} | val relL2={val_rel:.4e} | lr={lr_now:.2e}")

        with open(log_path, "a") as f:
            f.write(f"{epoch},{train_rel},{val_rel},{lr_now}\n")

        if val_rel < best_val:
            best_val = val_rel
            torch.save(
                {"model": model.state_dict(), "cfg": cfg.__dict__},
                best_path,
            )

    # Plot train/val relative L2 curve
    save_curve(
        train_rel_hist, val_rel_hist,
        out_path=results_part2_dir / "relL2_curve_task4_from_scratch.png",
        title="Task 4 (from scratch): Training/Validation relative L2",
        ylabel="Relative L2",
        yscale="linear",
    )

    # Load best checkpoint and evaluate on unknown test at t=1
    best_ckpt = torch.load(best_path, map_location=cfg.device)
    model.load_state_dict(best_ckpt["model"])
    model.eval()

    test_rel = eval_unknown_test_t1(model, test_path, batch_size=cfg.batch_size, device=cfg.device)

    print("\n===== TASK 4 FROM SCRATCH: TEST ON UNKNOWN (t=1.0) =====")
    print(f"Average relative L2 error (unknown test, t=1): {test_rel:.6f} ({100*test_rel:.2f}%)")

    out_test = results_dir / "test_result_unknown_t1.txt"
    with open(out_test, "w") as f:
        f.write(f"AvgRelativeL2_unknown_test_t1: {test_rel:.8f}\n")

    print(f"\nSaved best checkpoint to: {best_path}")
    print(f"Saved relL2 curve to: {results_part2_dir / 'relL2_curve_task4_from_scratch.png'}")
    print(f"Saved test result to: {out_test}")


if __name__ == "__main__":
    main()
