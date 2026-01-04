"""
Graph building utilities for variable coordinate datasets.
Handles neighbor computation and graph construction for VX mode.

MODIFIED for Strategy II (Random Sampling Tokenization):
- Supports per-sample latent queries (tokens) sampled from the domain points.
- Keeps backward compatibility with Strategy I (fixed shared latent_queries).
- Optionally supports resampling every epoch via seed = base_seed + 100000*epoch + sample_id.
"""
import time
import torch
from typing import List, Tuple, Optional

from ..model.layers.utils.neighbor_search import NeighborSearch
from ..utils.scaling import rescale

import os
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import numpy as np

def plot_radii_scatter(coords, radii, title, save_path, domain=None, max_circles=300):
    # coords: [N,2], radii: [N]
    if hasattr(coords, "detach"):
        coords = coords.detach().cpu().numpy()
    if hasattr(radii, "detach"):
        radii = radii.detach().cpu().numpy()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    N = coords.shape[0]
    # optional: subsample to avoid heavy plots
    if N > max_circles:
        idx = np.random.choice(N, size=max_circles, replace=False)
        coords = coords[idx]
        radii = radii[idx]

    fig, ax = plt.subplots(figsize=(5, 4))

    patches = [Circle((coords[i, 0], coords[i, 1]), radius=float(radii[i])) for i in range(coords.shape[0])]
    coll = PatchCollection(patches, array=radii, alpha=0.35)  # alpha for overlap visibility
    ax.add_collection(coll)
    plt.colorbar(coll, ax=ax, label="radius")

    ax.scatter(coords[:, 0], coords[:, 1], s=5)  # centers
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")

    if domain is not None:
        (xmin, ymin), (xmax, ymax) = domain
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
    else:
        ax.autoscale()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def sample_latent_queries_from_domain(
    x_coord_scaled: torch.Tensor,  # [n_nodes, coord_dim] in [-1,1]
    num_tokens: int,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Sample N points among the physical nodes as latent token coordinates."""
    n = x_coord_scaled.shape[0]
    if num_tokens >= n:
        return x_coord_scaled

    g = None
    if seed is not None:
        g = torch.Generator(device=x_coord_scaled.device)
        g.manual_seed(int(seed))

    idx = torch.randperm(n, generator=g, device=x_coord_scaled.device)[:num_tokens]
    return x_coord_scaled[idx]


def kth_nn_distance(points: torch.Tensor, k: int) -> torch.Tensor:
    """
    points: [N, D]
    Returns d_k for each point: distance to its k-th nearest neighbor (excluding itself).
    Output: [N]
    """
    N = points.size(0)
    if N <= 1:
        return torch.zeros(N, device=points.device, dtype=points.dtype)

    k_eff = min(k, N - 1)
    with torch.no_grad():
        d = torch.cdist(points, points)  # [N, N]
        d.fill_diagonal_(float("inf"))
        # take k-th smallest
        dk = torch.kthvalue(d, k_eff, dim=1).values  # [N]
    return dk


def kth_nn_distance_to_set(queries: torch.Tensor, data: torch.Tensor, k: int) -> torch.Tensor:
    """
    queries: [M, D], data: [N, D]
    Returns for each query: distance to k-th nearest in data.
    Output: [M]
    """
    M, N = queries.size(0), data.size(0)
    if M == 0 or N == 0:
        return torch.zeros(M, device=queries.device, dtype=queries.dtype)

    k_eff = min(k, N)
    with torch.no_grad():
        d = torch.cdist(queries, data)   # [M, N]
        dk = torch.kthvalue(d, k_eff, dim=1).values
    return dk

def coverage_ratio_from_csr(nbrs, num_phys):
    covered = torch.unique(nbrs["neighbors_index"]).numel()
    return covered / num_phys, covered

def degree_stats(nbrs):
    rs = nbrs["neighbors_row_splits"]
    deg = (rs[1:] - rs[:-1]).float()
    return deg.min().item(), deg.mean().item(), deg.max().item()

class GraphBuilder:
    """
    Builds encoder and decoder graphs for variable coordinate datasets.
    Handles neighbor computation with multiple radius scales.
    """

    def __init__(self, neighbor_search_method: str = "auto"):
        """
        Initialize graph builder.

        Args:
            neighbor_search_method: Method for neighbor search
        """
        self.nb_search = NeighborSearch(neighbor_search_method)



    def build_graphs_for_split(
        self,
        x_data: torch.Tensor,
        latent_queries: torch.Tensor,
        gno_radius: float,
        scales: List[float],
        tokenization: str = "grid",               # "grid" | "random"
        num_latent_tokens: int = 256,             # used if tokenization == "random"
        seed: int = 42,
        epoch: Optional[int] = None,
        resample_every_epoch: bool = False,
        coord_transform=None,
        dynamic_radius: bool = False,
        dynamic_radius_k: int = 8,
        dynamic_radius_alpha: float = 1.5,
        debug_plot_radii: bool = True,
        debug_plot_dir: str = ".results/debug",
        debug_plot_domain=None,

    ) -> Tuple[List, List, List]:
        """
        Build encoder and decoder graphs for a data split.

        Args:
            x_data: Coordinate data [n_samples, n_nodes, coord_dim] or [n_samples, 1, n_nodes, coord_dim]
            latent_queries: Latent query coordinates [n_latent, coord_dim] (used if tokenization == "grid")
            gno_radius: Base radius for neighbor search
            scales: List of scale factors for multi-scale graphs
            tokenization: "grid" (Strategy I) or "random" (Strategy II)
            num_latent_tokens: number of randomly sampled latent tokens (Strategy II)
            seed: base seed for reproducibility
            epoch: current epoch index (only used if resample_every_epoch=True)
            resample_every_epoch: if True, change sampled tokens each epoch

        Returns:
            tuple: (encoder_graphs_list, decoder_graphs_list, latent_queries_per_sample)
        """
        print(f"Building graphs for {len(x_data)} samples...")
        start_time = time.time()

        encoder_graphs = []
        decoder_graphs = []
        latent_queries_per_sample = []

        for i, x_sample in enumerate(x_data):
            # Handle different input shapes
            if x_sample.dim() == 3 and x_sample.shape[0] == 1:
                # Shape: [1, n_nodes, coord_dim] -> [n_nodes, coord_dim]
                x_coord = x_sample[0]
            elif x_sample.dim() == 2:
                # Shape: [n_nodes, coord_dim]
                x_coord = x_sample
            else:
                raise ValueError(f"Unexpected coordinate shape: {x_sample.shape}")

            # Rescale coordinates to [-1, 1] range
            # x_coord_scaled = rescale(x_coord, (-1, 1))
    
            if coord_transform is not None:
                x_coord_scaled = coord_transform(x_coord)   
            else:
                x_coord_scaled = rescale(x_coord, (-1, 1))  

            

            # Choose latent queries (tokens) for this sample
            if tokenization == "grid":
                latent_q = latent_queries  # shared across samples (Strategy I)
            elif tokenization == "random":
                # Fixed tokens: seed depends on sample id
                # Resample: seed depends on epoch as well
                if resample_every_epoch and epoch is not None:
                    local_seed = seed + 100000 * epoch + i
                else:
                    local_seed = seed + i

                latent_q = sample_latent_queries_from_domain(
                    x_coord_scaled, num_latent_tokens, seed=local_seed
                )
            else:
                raise ValueError(f"Unknown tokenization strategy: {tokenization}")

            latent_queries_per_sample.append(latent_q)

            if dynamic_radius:
                dk_latent = kth_nn_distance(latent_q, dynamic_radius_k)                # [N_latent]
                r_latent = dynamic_radius_alpha * dk_latent

                dk_phys = kth_nn_distance_to_set(x_coord_scaled, latent_q, dynamic_radius_k)  # [N_phys]
                r_phys = dynamic_radius_alpha * dk_phys

                # avoid degenerate 0 radii (rare but possible)
                eps = 1e-8
                r_latent = torch.clamp(r_latent, min=eps)
                r_phys = torch.clamp(r_phys, min=eps)

                #debug plot (only for first sample to avoid spamming)
                if debug_plot_radii and i == 0:
                    print("[R] r_latent min/mean/max:", r_latent.min().item(), r_latent.mean().item(), r_latent.max().item())
                    plot_radii_scatter(
                        coords=latent_q,
                        radii=r_latent,
                        title=f"Dynamic radius (latent tokens) - split sample {i}",
                        save_path=os.path.join(debug_plot_dir, "radii_latent_sample0.png"),
                        domain=debug_plot_domain,
                    )
                    plot_radii_scatter(
                        coords=x_coord_scaled,
                        radii=r_phys,
                        title=f"Dynamic radius (physical queries) - split sample {i}",
                        save_path=os.path.join(debug_plot_dir, "radii_phys_sample0.png"),
                        domain=debug_plot_domain,
                    )
            else:
                r_latent = None
                r_phys = None

            #encoder graphs (physical -> latent)
            encoder_nbrs_sample = []
            for scale in scales:
                # scaled_radius = gno_radius * scale
                # with torch.no_grad():
                #     nbrs = self.nb_search(x_coord_scaled, latent_q, scaled_radius)

                if dynamic_radius:
                    radius_vec = r_latent * scale
                    nbrs = self.nb_search(x_coord_scaled, latent_q, radius_vec)
                else:
                    scaled_radius = gno_radius * scale
                    nbrs = self.nb_search(x_coord_scaled, latent_q, scaled_radius)


                encoder_nbrs_sample.append(nbrs)



            if dynamic_radius and i == 0:
                num_phys = x_coord_scaled.shape[0]
                for s_idx, nbrs_s in enumerate(encoder_nbrs_sample):
                    ratio, covered = coverage_ratio_from_csr(nbrs_s, num_phys)
                    print(f"[COVERAGE] encoder scale {s_idx}: {covered}/{num_phys} = {ratio:.4f}")

            if dynamic_radius and i == 0:
                for s_idx, nbrs_s in enumerate(encoder_nbrs_sample):
                    dmin, dmean, dmax = degree_stats(nbrs_s)
                    print(f"[DEGREE] encoder scale {s_idx}: min={dmin:.0f}, mean={dmean:.1f}, max={dmax:.0f}")


            encoder_graphs.append(encoder_nbrs_sample)



            #decoder graphs (latent -> physical)
            decoder_nbrs_sample = []
            for scale in scales:
                # scaled_radius = gno_radius * scale
                # with torch.no_grad():
                #     nbrs = self.nb_search(latent_q, x_coord_scaled, scaled_radius)

                if dynamic_radius:
                    radius_vec = r_phys * scale
                    nbrs = self.nb_search(latent_q, x_coord_scaled, radius_vec)
                else:
                    scaled_radius = gno_radius * scale
                    nbrs = self.nb_search(latent_q, x_coord_scaled, scaled_radius)

                decoder_nbrs_sample.append(nbrs)
            decoder_graphs.append(decoder_nbrs_sample)

            if (i + 1) % 100 == 0 or i == len(x_data) - 1:
                elapsed = time.time() - start_time
                print(f"Processed {i + 1}/{len(x_data)} samples ({elapsed:.2f}s)")

        total_time = time.time() - start_time
        print(f"Graph building completed in {total_time:.2f}s")

        return encoder_graphs, decoder_graphs, latent_queries_per_sample

    def build_all_graphs(
        self,
        data_splits: dict,
        latent_queries: torch.Tensor,
        gno_radius: float,
        scales: List[float],
        build_train: bool = True,
        tokenization: str = "grid",
        num_latent_tokens: int = 256,
        seed: int = 42,
        epoch: Optional[int] = None,
        resample_every_epoch: bool = False,
        coord_transform=None,
        dynamic_radius: bool = False,
        dynamic_radius_k: int = 8,
        dynamic_radius_alpha: float = 1.5,
        debug_plot_radii: bool = True,
        debug_plot_dir: str = ".results/debug",
        debug_plot_domain=None,
    ) -> dict:
        """
        Build graphs for all data splits.

        Args:
            data_splits: Dictionary with train/val/test splits
            latent_queries: Latent query coordinates (used if tokenization == "grid")
            gno_radius: Base radius for neighbor search
            scales: Scale factors for multi-scale graphs
            build_train: Whether to build train/val graphs (skip if testing only)
            tokenization: "grid" or "random"
            num_latent_tokens: number of sampled latent tokens if random
            seed: base seed
            epoch: epoch index if resampling
            resample_every_epoch: whether to resample tokens each epoch

        Returns:
            dict: Dictionary with encoder/decoder graphs (+ latent_queries) for each split
        """
        all_graphs = {}

        if "test" in data_splits:
            encoder_test, decoder_test, latent_test = self.build_graphs_for_split(
                data_splits["test"]["x"],
                latent_queries,
                gno_radius,
                scales,
                tokenization=tokenization,
                num_latent_tokens=num_latent_tokens,
                seed=seed,
                epoch=epoch,
                resample_every_epoch=resample_every_epoch,
                coord_transform=coord_transform,
  
                dynamic_radius=dynamic_radius,
                dynamic_radius_k=dynamic_radius_k,
                dynamic_radius_alpha=dynamic_radius_alpha,
                debug_plot_radii=debug_plot_radii,
                debug_plot_dir=debug_plot_dir,
                debug_plot_domain=debug_plot_domain,
            )
            all_graphs["test"] = {"encoder": encoder_test, "decoder": decoder_test, "latent_queries": latent_test}

        if build_train:
            if "train" in data_splits:
                encoder_train, decoder_train, latent_train = self.build_graphs_for_split(
                    data_splits["train"]["x"],
                    latent_queries,
                    gno_radius,
                    scales,
                    tokenization=tokenization,
                    num_latent_tokens=num_latent_tokens,
                    seed=seed,
                    epoch=epoch,
                    resample_every_epoch=resample_every_epoch,
                    coord_transform=coord_transform,
                    dynamic_radius=dynamic_radius,
                    dynamic_radius_k=dynamic_radius_k,
                    dynamic_radius_alpha=dynamic_radius_alpha,
                    debug_plot_radii=debug_plot_radii,
                    debug_plot_dir=debug_plot_dir,
                    debug_plot_domain=debug_plot_domain,
                )
                all_graphs["train"] = {"encoder": encoder_train, "decoder": decoder_train, "latent_queries": latent_train}

            if "val" in data_splits:
                encoder_val, decoder_val, latent_val = self.build_graphs_for_split(
                    data_splits["val"]["x"],
                    latent_queries,
                    gno_radius,
                    scales,
                    tokenization=tokenization,
                    num_latent_tokens=num_latent_tokens,
                    seed=seed,
                    epoch=epoch,
                    resample_every_epoch=resample_every_epoch,
                    coord_transform=coord_transform,
                    dynamic_radius=dynamic_radius,
                    dynamic_radius_k=dynamic_radius_k,
                    dynamic_radius_alpha=dynamic_radius_alpha,
                    debug_plot_radii=debug_plot_radii,
                    debug_plot_dir=debug_plot_dir,
                    debug_plot_domain=debug_plot_domain,
                )
                all_graphs["val"] = {"encoder": encoder_val, "decoder": decoder_val, "latent_queries": latent_val}
        else:
            all_graphs["train"] = None
            all_graphs["val"] = None

        return all_graphs

    def validate_graphs(self, graphs: dict, expected_samples: dict):
        """
        Validate that graphs have correct structure and sizes.

        Args:
            graphs: Graph dictionary
            expected_samples: Expected number of samples per split
        """
        for split_name, split_graphs in graphs.items():
            if split_graphs is None:
                continue

            encoder_graphs = split_graphs["encoder"]
            decoder_graphs = split_graphs["decoder"]
            expected_count = expected_samples.get(split_name, 0)

            assert (
                len(encoder_graphs) == expected_count
            ), f"Encoder graphs for {split_name}: expected {expected_count}, got {len(encoder_graphs)}"
            assert (
                len(decoder_graphs) == expected_count
            ), f"Decoder graphs for {split_name}: expected {expected_count}, got {len(decoder_graphs)}"

            # Validate individual samples
            for i, (enc_sample, dec_sample) in enumerate(zip(encoder_graphs, decoder_graphs)):
                assert isinstance(enc_sample, list), f"Encoder sample {i} should be list of scales"
                assert isinstance(dec_sample, list), f"Decoder sample {i} should be list of scales"
                assert len(enc_sample) == len(dec_sample), (
                    f"Encoder and decoder should have same number of scales for sample {i}"
                )

            # Optional: validate latent queries list length if present
            if "latent_queries" in split_graphs:
                assert len(split_graphs["latent_queries"]) == expected_count, (
                    f"Latent queries for {split_name}: expected {expected_count}, got {len(split_graphs['latent_queries'])}"
                )

        print("Graph validation passed")


class CachedGraphBuilder(GraphBuilder):
    """
    Graph builder with caching capabilities.
    Can save and load pre-computed graphs to avoid recomputation.
    """

    def __init__(self, neighbor_search_method: str = "auto", cache_dir: Optional[str] = None):
        super().__init__(neighbor_search_method)
        self.cache_dir = cache_dir

    def _get_cache_path(self, dataset_name: str, split_name: str, graph_type: str) -> str:
        """Get cache file path for graphs."""
        if self.cache_dir is None:
            raise ValueError("Cache directory not specified")

        import os

        os.makedirs(self.cache_dir, exist_ok=True)
        return os.path.join(self.cache_dir, f"{dataset_name}_{split_name}_{graph_type}_graphs.pt")

    def save_graphs(self, graphs: dict, dataset_name: str):
        """Save graphs to cache."""
        if self.cache_dir is None:
            print("No cache directory specified, skipping graph save")
            return

        for split_name, split_graphs in graphs.items():
            if split_graphs is None:
                continue

            # Save encoder graphs
            encoder_path = self._get_cache_path(dataset_name, split_name, "encoder")
            torch.save(split_graphs["encoder"], encoder_path)

            # Save decoder graphs
            decoder_path = self._get_cache_path(dataset_name, split_name, "decoder")
            torch.save(split_graphs["decoder"], decoder_path)

            # Save latent queries (needed for Strategy II)
            if "latent_queries" in split_graphs:
                latent_path = self._get_cache_path(dataset_name, split_name, "latent_queries")
                torch.save(split_graphs["latent_queries"], latent_path)

        print(f"Graphs saved to cache directory: {self.cache_dir}")

    def load_graphs(self, dataset_name: str, splits: List[str]) -> Optional[dict]:
        """Load graphs from cache."""
        if self.cache_dir is None:
            return None

        try:
            all_graphs = {}
            for split_name in splits:
                encoder_path = self._get_cache_path(dataset_name, split_name, "encoder")
                decoder_path = self._get_cache_path(dataset_name, split_name, "decoder")
                latent_path = self._get_cache_path(dataset_name, split_name, "latent_queries")

                import os

                if not (os.path.exists(encoder_path) and os.path.exists(decoder_path)):
                    print(f"Cache files not found for split: {split_name}")
                    return None

                encoder_graphs = torch.load(encoder_path)
                decoder_graphs = torch.load(decoder_path)

                split_dict = {
                    "encoder": encoder_graphs,
                    "decoder": decoder_graphs,
                }

                # latent queries optional (Strategy II)
                if os.path.exists(latent_path):
                    split_dict["latent_queries"] = torch.load(latent_path)

                all_graphs[split_name] = split_dict

            print(f"Graphs loaded from cache directory: {self.cache_dir}")
            return all_graphs

        except Exception as e:
            print(f"Failed to load graphs from cache: {e}")
            return None

    def build_all_graphs(
        self,
        data_splits: dict,
        latent_queries: torch.Tensor,
        gno_radius: float,
        scales: List[float],
        dataset_name: str = "dataset",
        build_train: bool = True,
        use_cache: bool = True,
        tokenization: str = "grid",
        num_latent_tokens: int = 256,
        seed: int = 42,
        epoch: Optional[int] = None,
        resample_every_epoch: bool = False,

        dynamic_radius: bool = False,

        dynamic_radius_k: int = 8,

        dynamic_radius_alpha: float = 1.5,

        debug_plot_radii: bool = True,

        debug_plot_dir: str = ".results/debug",

        debug_plot_domain = None
    ) -> dict:
        """
        Build graphs with caching support.

        Args:
            data_splits: Data splits dictionary
            latent_queries: Latent query coordinates
            gno_radius: Base radius
            scales: Scale factors
            dataset_name: Name for cache files
            build_train: Whether to build train/val graphs
            use_cache: Whether to use cached graphs
            tokenization: "grid" or "random"
            num_latent_tokens: sampled tokens if random
            seed: base seed
            epoch: epoch for resampling
            resample_every_epoch: whether to resample each epoch

        Returns:
            dict: Graph dictionary
        """
        # Try to load from cache first
        if use_cache and self.cache_dir is not None:
            cache_splits = ["test"]
            if build_train:
                cache_splits.extend(["train", "val"])

            cached_graphs = self.load_graphs(dataset_name, cache_splits)
            if cached_graphs is not None:
                return cached_graphs

        # Build graphs if not cached
        graphs = super().build_all_graphs(
            data_splits,
            latent_queries,
            gno_radius,
            scales,
            build_train=build_train,
            tokenization=tokenization,
            num_latent_tokens=num_latent_tokens,
            seed=seed,
            epoch=epoch,
            resample_every_epoch=resample_every_epoch,
        )

        # Save to cache
        if use_cache:
            self.save_graphs(graphs, dataset_name)

        return graphs
