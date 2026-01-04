from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt



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
        # x: (B, in_ch, S)
        x_ft = torch.fft.rfft(x)  # (B, in_ch, S//2+1)
        modes = min(self.modes1, x_ft.size(-1))

        out_ft = torch.zeros(
            (x.size(0), self.out_channels, x_ft.size(-1)),
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, :modes] = self.compl_mul1d(x_ft[:, :, :modes], self.weights1[:, :, :modes])
        x = torch.fft.irfft(out_ft, n=x.shape[-1], dim=-1)  # (B, out_ch, S)
        return x


class FNO1d(nn.Module):
    """
    Input:  (B, S, 2) = [u0(x), x]
    Output: (B, S, 1) = u(t=1)(x)
    """
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
        x = self.linear_p(x)      # (B,S,C)
        x = x.permute(0, 2, 1)    # (B,C,S)

        x = self._fno_block(x, self.spect1, self.lin0)
        x = self._fno_block(x, self.spect2, self.lin1)
        x = self._fno_block(x, self.spect3, self.lin2)
        x = self._fno_block(x, self.spect4, self.lin3)

        x = x.permute(0, 2, 1)    # (B,S,C)
        x = self.act(self.linear_q(x))
        x = self.output_layer(x)  # (B,S,1)
        return x


@torch.no_grad()
def evaluate_file(model: nn.Module, npy_path: Path, batch_size: int, device: str) -> float:
    """
    npy file shape: (N, 5, S)
    Uses u0 = [:,0,:], uT = [:,4,:] at t=1.
    Returns average relative L2 over N trajectories.
    """
    arr = np.load(npy_path)  # (N,5,S)
    assert arr.ndim == 3 and arr.shape[1] == 5, f"Bad shape {arr.shape} in {npy_path.name}"
    N, _, S = arr.shape

    u0 = torch.from_numpy(arr[:, 0, :]).float()  # (N,S)
    uT = torch.from_numpy(arr[:, 4, :]).float()  # (N,S)

    #grid for this resolution
    x = torch.linspace(0.0, 1.0, S).float()[None, :].repeat(N, 1)  # (N,S)

    inp = torch.stack([u0, x], dim=-1)
    tgt = uT[:, :, None]

    model.eval()
    err_sum = 0.0
    n_done = 0

    for i in range(0, N, batch_size):
        inp_b = inp[i:i + batch_size].to(device)
        tgt_b = tgt[i:i + batch_size].to(device)

        pred_b = model(inp_b)                 # (B,S,1)
        err_b = relative_l2(pred_b, tgt_b)    # (B,)
        err_sum += err_b.sum().item()
        n_done += err_b.numel()

    return err_sum / n_done


def save_resolution_barplot(results: dict, out_path: Path) -> None:
    """
    results: {resolution: avg_error} where avg_error is in [0,1]
    Saves a barplot in percentage (%).
    """
    resolutions = sorted(results.keys())
    errors_pct = [100.0 * results[r] for r in resolutions]

    plt.figure(figsize=(6, 4))
    plt.bar([str(r) for r in resolutions], errors_pct)
    plt.xlabel("Resolution (s)")
    plt.ylabel("Average relative L2 error (%)")
    plt.title("Task 2: Average relative L2 error across resolutions (t=1)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    data_dir = Path("data/part2/FNO_data")
    ckpt_path = Path("results/part2/task1_one2one/best_model.pt")

    out_csv = Path("results/part2/task2_resolution_results.txt")
    out_plot = Path("results/part2/task2_resolution_barplot.png")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("cfg", {})
    modes = int(cfg.get("modes", 16))
    width = int(cfg.get("width", 64))

    model = FNO1d(modes=modes, width=width).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    batch_size = 64
    resolutions = [32, 64, 96, 128]

    results = {}
    print("===== TASK 2: Resolution generalization =====")
    for s in resolutions:
        npy_path = data_dir / f"data_test_{s}.npy"
        err = evaluate_file(model, npy_path, batch_size=batch_size, device=device)
        results[s] = err
        print(f"data_test_{s}.npy: avg relative L2 = {err:.6f}")


    with open(out_csv, "w") as f:
        f.write("Resolution,AvgRelativeL2\n")
        for s in sorted(results.keys()):
            f.write(f"{s},{results[s]:.8f}\n")

    save_resolution_barplot(results, out_plot)

    print(f"\nSaved results table to: {out_csv}")
    print(f"Saved barplot to: {out_plot}")


if __name__ == "__main__":
    main()
