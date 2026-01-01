import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Simple MLP u_theta: R^2 -> R with Tanh activations.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_layers: int, width: int, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)

        layers = [nn.Linear(in_dim, width), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)

        self._init_xavier_tanh()

    def _init_xavier_tanh(self) -> None:
        gain = nn.init.calculate_gain("tanh")

        def init_fn(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=gain)
                nn.init.zeros_(m.bias)

        self.apply(init_fn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
