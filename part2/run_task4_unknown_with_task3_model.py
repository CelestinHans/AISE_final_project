from pathlib import Path
import re
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def relative_l2(pred: torch.Tensor, true: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    err_n = ||pred-true||_2 / ||true||_2  (per trajectory)
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
    Input:  (B,S,3) = [u(t_i)(x), x, Δt]
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
        x = self.linear_p(inp)  # (B,S,C)
        x = x.permute(0, 2, 1)  # (B,C,S)

        x = self._block(x, dt, self.spect1, self.lin0, self.film1)
        x = self._block(x, dt, self.spect2, self.lin1, self.film2)
        x = self._block(x, dt, self.spect3, self.lin2, self.film3)
        x = self._block(x, dt, self.spect4, self.lin3, self.film4)

        x = x.permute(0, 2, 1)  # (B,S,C)
        x = self.act(self.linear_q(x))
        x = self.output_layer(x)  # (B,S,1)
        return x




@torch.no_grad()
def predict_unknown_t1(model: nn.Module, npy_path: Path, batch_size: int, device: str):

    arr = np.load(npy_path)  # (N,5,128)
    assert arr.ndim == 3 and arr.shape[1] == 5, f"Bad shape {arr.shape} in {npy_path.name}"
    N, _, S = arr.shape

    u0 = torch.from_numpy(arr[:, 0, :]).float()  # (N,S)
    u1 = torch.from_numpy(arr[:, 4, :]).float()  # (N,S)

    x = torch.linspace(0.0, 1.0, S).float()[None, :].repeat(N, 1)  # (N,S)
    dt = torch.full((N, S), 1.0).float()                           # (N,S)

    inp = torch.stack([u0, x, dt], dim=-1)     # (N,S,3)
    tgt = u1[:, :, None]                      # (N,S,1)

    model.eval()

    preds_list = []
    trues_list = []
    errs_list = []

    for i in range(0, N, batch_size):
        inp_b = inp[i:i+batch_size].to(device)
        tgt_b = tgt[i:i+batch_size].to(device)

        pred_b = model(inp_b)                 # (B,S,1)
        err_b = relative_l2(pred_b, tgt_b)    # (B,)

        preds_list.append(pred_b.squeeze(-1).detach().cpu())
        trues_list.append(tgt_b.squeeze(-1).detach().cpu())
        errs_list.append(err_b.detach().cpu())

    preds = torch.cat(preds_list, dim=0)   # (N,S)
    trues = torch.cat(trues_list, dim=0)   # (N,S)
    per_err = torch.cat(errs_list, dim=0).numpy()  # (N,)
    avg_rel = float(per_err.mean())
    return avg_rel, preds, trues, per_err


def save_single_plot(x, pred, true, out_path: Path):
    plt.figure()
    plt.plot(x, true, label="true u(t=1)")
    plt.plot(x, pred, label="pred u(t=1)")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.title("Unknown test: prediction vs true at t=1")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_mean_plot(x, pred_mean, true_mean, out_path: Path):
    plt.figure()
    plt.plot(x, true_mean, label="mean true u(t=1)")
    plt.plot(x, pred_mean, label="mean pred u(t=1)")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.title("Unknown test: mean prediction vs mean true at t=1")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_dir = Path("data/part2/FNO_data")
    unknown_test_path = data_dir / "data_test_unknown_128.npy"

    task3_ckpt_path = Path("results/part2/task3_all2all_time_dependent/best_model.pt")


    out_dir = Path("results/part2")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_txt = out_dir / "task4_unknown_test_t1_result.txt"
    out_single = out_dir / "task4_unknown_single_trajectory_t1.png"
    out_mean = out_dir / "task4_unknown_mean_trajectory_t1.png"

  
    ckpt = torch.load(task3_ckpt_path, map_location=device)
    cfg = ckpt.get("cfg", {})
    modes = int(cfg.get("modes", 16))
    width = int(cfg.get("width", 64))
    use_bn = bool(cfg.get("use_bn", True))

    model = TimeDependentFNO1d_All2All(modes=modes, width=width, use_bn=use_bn).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    avg_err, preds, trues, per_err = predict_unknown_t1(
        model, unknown_test_path, batch_size=64, device=device
    )


   

    print("===== TASK 4: Zero-shot on unknown distribution at t=1.0 =====")
    print(f"Task 4 (Task 3 model) avg relative L2 on data_test_unknown_128.npy: {avg_err:.6f} ({100*avg_err:.2f}%)")
                            {avg_err - task1_err:+.6f}")

    with open(out_txt, "w") as f:
        f.write(f"Task4_avg_relL2_unknown_test128_t1: {avg_err:.8f}\n")
        

    S = preds.shape[1]
    x = np.linspace(0.0, 1.0, S)

    idx = 0  # choose the first trajectory by default
    save_single_plot(x, preds[idx].numpy(), trues[idx].numpy(), out_single)
    save_mean_plot(x, preds.mean(dim=0).numpy(), trues.mean(dim=0).numpy(), out_mean)

    print(f"\nSaved report to: {out_txt}")
    print(f"Saved single-trajectory plot to: {out_single}")
    print(f"Saved mean-trajectory plot to: {out_mean}")


if __name__ == "__main__":
    main()
