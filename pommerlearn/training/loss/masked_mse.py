import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class MaskedMSELoss(_Loss):
    """
    This criterion describes a cross entropy loss with continious values (non-one-hot-encoded) as targets.
    """
    def __init__(self) -> None:
        super(MaskedMSELoss, self).__init__()

    def forward(self, input: Tensor, target: Tensor, mask: Tensor = None) -> Tensor:
        if mask is None:
            return F.mse_loss(input, target)
        else:
            return torch.sum(torch.pow(input - target, 2) * mask) / torch.sum(mask)
