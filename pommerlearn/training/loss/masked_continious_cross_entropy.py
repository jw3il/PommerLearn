import torch
from torch.nn.modules.loss import _Loss
from torch import Tensor
import torch.nn.functional as F


class MaskedContiniousCrossEntropyLoss(_Loss):
    """
    This criterion describes a cross entropy loss with continious values (non-one-hot-encoded) as targets.

    Examples::

        >>> loss = MaskedContiniousCrossEntropyLoss()
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.randn(3, 5)
        >>> output = loss(input, target)
        >>> output.backward()
    """
    def __init__(self) -> None:
        super(MaskedContiniousCrossEntropyLoss, self).__init__()

    def forward(self, log_input: Tensor, target: Tensor, mask: Tensor = None) -> Tensor:
        if mask is None:
            return -(target * F.log_softmax(log_input, dim=-1)).sum(dim=-1).mean()
        else:
            assert len(log_input.shape) == 3, "Masking is currently only supported for sequences"

            cross_entropy = -(target * F.log_softmax(log_input, dim=-1) * mask).sum()
            return cross_entropy / mask.sum()
