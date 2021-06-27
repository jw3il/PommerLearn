from torch.nn.modules.loss import _Loss
from torch import Tensor
import torch
import torch.nn.functional as F

from training.loss.masked_continious_cross_entropy import MaskedContiniousCrossEntropyLoss


class MaskedKLDivergenceLoss(_Loss):
    """
    KL Divergence with masking (e.g. for comparing sequences).
    """
    def __init__(self) -> None:
        super(MaskedKLDivergenceLoss, self).__init__()
        self.masked_cross_entropy = MaskedContiniousCrossEntropyLoss()

    def forward(self, log_input: Tensor, target: Tensor, mask: Tensor = None) -> Tensor:
        if mask is None:
            cross_entropy = -(target * F.log_softmax(log_input, dim=-1)).sum(dim=-1).mean()
            # nan_to_num converts nan to 0 (required because when target = 0, we get 0 * inf = nan)
            target_entropy = -torch.nan_to_num(target * torch.log(target)).sum(dim=-1).mean()

            return cross_entropy - target_entropy
        else:
            assert len(log_input.shape) == 3, "Masking is currently only supported for sequences"

            cross_entropy = -(target * F.log_softmax(log_input, dim=-1) * mask).sum()
            target_entropy = -(torch.nan_to_num(target * target.log()) * mask).sum()

            return (cross_entropy - target_entropy) / mask.sum()
