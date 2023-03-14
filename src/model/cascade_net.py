"""Cascaded Net"""

import torch

from config_manager.manager import Params
from model.layers import PositionalEncoding
from model.unet import UNet


class CascadeNet(torch.nn.Module):
    """Cascaded Net architecture
    Args:
        params: Hyper-parameters
    """

    def __init__(
        self,
        params: Params,
    ) -> None:
        super().__init__()
        num_cascades = len(params.out_channels)
        nets = [
            UNet(
                1 + 2 * params.num_encodings,
                params.out_channels[i],
                params.filters[i],
                params.kernels[i],
            )
            for i in range(num_cascades)
        ]
        self.net_stack = torch.nn.ModuleList(nets)
        self.pec = PositionalEncoding(params.num_encodings)

    def forward(self, x_t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward propagation"""
        for layer in self.net_stack:
            with torch.no_grad():
                x_t = self.pec(x_t)
            x_t = layer(x_t, mask)
        return x_t
