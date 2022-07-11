import torch
from torch.nn.modules.loss import _Loss
from torch import Tensor
import torch.nn.functional as F


class MaskedContinuousCrossEntropyLoss(_Loss):
    """
    This criterion describes a cross entropy loss with continious values (non-one-hot-encoded) as targets.

    Examples::

        >>> loss = MaskedContinuousCrossEntropyLoss()
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.randn(3, 5)
        >>> output = loss(input, target)
        >>> output.backward()
    """
    def __init__(self, argmax_target: bool = False) -> None:
        """
        :param argmax_target: Along dim -1 of the target, whether to set the target entry with the highest probability
            to 1 and all the others to 0 instead of using the actual continuous target distribution.
        """
        super(MaskedContinuousCrossEntropyLoss, self).__init__()
        self.argmax_target = argmax_target

    def forward(self, log_input: Tensor, target: Tensor, mask: Tensor = None) -> Tensor:
        """
        Calculates the loss.

        :param log_input: log of input (predicted) probabilities, e.g. model output
        :param target: target probabilities
        :param mask: allows to mask samples in sequences
        :return:
        """
        if self.argmax_target:
            max_indices = torch.max(target, dim=-1)[1].unsqueeze(-1)
            target = torch.zeros_like(target, device=target.device)
            target.scatter_(-1, max_indices, 1.0)

        if mask is None:
            return -(target * F.log_softmax(log_input, dim=-1)).sum(dim=-1).mean()
        else:
            assert len(log_input.shape) == 3, "Masking is currently only supported for sequences"

            cross_entropy = -(target * F.log_softmax(log_input, dim=-1) * mask).sum()
            return cross_entropy / mask.sum()
