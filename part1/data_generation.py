import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def generate_grid(n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a uniform 2D grid on [0,1]x[0,1].

    Args:
        n: Number of points per dimension.

    Returns:
        Meshgrid (X, Y) of shape (n, n).
    """
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    return np.meshgrid(x, y, indexing="ij")


def sample_fourier_coefficients(k: int, seed: int | None = None) -> np.ndarray:
    """
    Sample Gaussian Fourier coefficients a_{ij}.

    Args:
        k: Maximum frequency.
        seed: Optional random seed.

    Returns:
        Array of shape (k, k).
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal((k, k))


def generate_rhs(
    X: np.ndarray, Y: np.ndarray, coeffs: np.ndarray, r: float
) -> np.ndarray:
    """
    Generate the right-hand side f(x,y).

    Args:
        X, Y: Grid coordinates.
        coeffs: Fourier coefficients a_{ij}.
        r: Decay parameter.

    Returns:
        f evaluated on the grid.
    """
    f = np.zeros_like(X)
    k = coeffs.shape[0]

    for i in range(1, k + 1):
        for j in range(1, k + 1):
            a = coeffs[i - 1, j - 1]
            f += a * np.sin(i * np.pi * X) * np.sin(j * np.pi * Y) * (
                i**2 + j**2
            ) ** r
    f = f*np.pi / k**2
    return f


def generate_solution(
    X: np.ndarray, Y: np.ndarray, coeffs: np.ndarray, r: float
) -> np.ndarray:
    """
    Generate the exact solution u(x,y) of the Poisson problem.

    Args:
        X, Y: Grid coordinates.
        coeffs: Fourier coefficients a_{ij}.
        r: Decay parameter.

    Returns:
        u evaluated on the grid.
    """
    u = np.zeros_like(X)
    k = coeffs.shape[0]

    for i in range(1, k + 1):
        for j in range(1, k + 1):
            a = coeffs[i - 1, j - 1]
            u += a * np.sin(i * np.pi * X) * np.sin(j * np.pi * Y) * (
                (i**2 + j**2) ** (r - 1) 
            )
    u=u / (np.pi*k**2)
    return u


def save_pair_plots(
    f: np.ndarray,
    u: np.ndarray,
    out_png: Path,
    title_f: str = "Source f",
    title_u: str = "Ground Truth u",
) -> None:
    """
    Save a side-by-side plot (f -> u) as a PNG.

    Args:
        f: Source field, shape (n, n).
        u: Solution field, shape (n, n).
        out_png: Output path for the PNG.
        title_f: Left title.
        title_u: Right title.
    """
    fig = plt.figure(figsize=(10, 4), dpi=150)
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 0.05, 0.18, 1, 0.05], wspace=0.25)

    ax_f = fig.add_subplot(gs[0, 0])
    ax_cb_f = fig.add_subplot(gs[0, 1])
    ax_mid = fig.add_subplot(gs[0, 2])
    ax_u = fig.add_subplot(gs[0, 3])
    ax_cb_u = fig.add_subplot(gs[0, 4])

    im_f = ax_f.imshow(f, origin="lower", aspect="auto", cmap="jet")
    ax_f.set_title(title_f)
    fig.colorbar(im_f, cax=ax_cb_f)

    ax_mid.axis("off")
    ax_mid.annotate(
        "",
        xy=(0.95, 0.5),
        xytext=(0.05, 0.5),
        arrowprops=dict(arrowstyle="simple", lw=0, color="black"),
        xycoords="axes fraction",
        textcoords="axes fraction",
    )

    im_u = ax_u.imshow(u, origin="lower", aspect="auto", cmap="jet")
    ax_u.set_title(title_u)
    fig.colorbar(im_u, cax=ax_cb_u)

    for ax in (ax_f, ax_u):
        ax.set_xlabel("")
        ax.set_ylabel("")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def generate_dataset_and_plots(
    n: int,
    ks: list[int],
    r: float,
    num_samples_per_k: int,
    num_plots_per_k: int,
    seed: int,
    data_root: Path,
    figs_root: Path,
) -> None:
    """
    Generate (f, u) pairs for multiple K values, save as .npy, and export PNG plots.

    Args:
        n: Grid resolution.
        ks: List of K values.
        r: Decay parameter.
        num_samples_per_k: Number of samples to generate for each K.
        num_plots_per_k: Number of PNG examples to save per K.
        seed: RNG seed.
        data_root: Root folder for .npy outputs.
        figs_root: Root folder for PNG plots.
    """
    rng = np.random.default_rng(seed)
    X, Y = generate_grid(n)

    for k in ks:
        out_data = data_root / f"K{k}"
        out_figs = figs_root / f"K{k}"
        out_data.mkdir(parents=True, exist_ok=True)
        out_figs.mkdir(parents=True, exist_ok=True)

        for idx in range(num_samples_per_k):
            coeffs = sample_fourier_coefficients(k, rng)
            f = generate_rhs(X, Y, coeffs, r)
            u = generate_solution(X, Y, coeffs, r)

            np.save(out_data / f"f_{idx}.npy", f)
            np.save(out_data / f"u_{idx}.npy", u)

            if idx < num_plots_per_k:
                save_pair_plots(
                    f,
                    u,
                    out_figs / f"example_{idx}.png",
                    title_f="Source f",
                    title_u="Ground Truth u",
                )


if __name__ == "__main__":
    generate_dataset_and_plots(
        n=64,
        ks=[1, 4, 8, 16],
        r=0.5,
        num_samples_per_k=100,
        num_plots_per_k=3,
        seed=0,
        data_root=Path("data/part1"),
        figs_root=Path("results/part1/figures"),
    )