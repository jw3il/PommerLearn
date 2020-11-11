from torch.nn.modules.loss import _WeightedLoss
from torch import Tensor
import torch.nn as nn


class CrossEntropyLossContinious(_WeightedLoss):
    r"""This criterion describes a cross entropy loss with continious values (non-one-hot-encoded) as targets.

    Examples::

        >>> loss = CrossEntropyLossContinious()
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.randn(3, 5)
        >>> output = loss(input, target)
        >>> output.backward()
    """
    def __init__(self) -> None:
        super(CrossEntropyLossContinious, self).__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return -(target * nn.LogSoftmax(input)).sum(dim=1).mean()
