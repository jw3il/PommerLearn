from torch.nn.modules.loss import _WeightedLoss
from torch import Tensor
import torch.nn.functional as F


class MaskedContiniousCrossEntropyLoss(_WeightedLoss):
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

    def forward(self, input: Tensor, target: Tensor, mask: Tensor=None) -> Tensor:
        if mask is None:
            return -(target * F.log_softmax(input, dim=-1)).sum(dim=-1).mean()
        else:
            return -(target * F.log_softmax(input, dim=-1) * mask).sum(dim=-1).mean()
