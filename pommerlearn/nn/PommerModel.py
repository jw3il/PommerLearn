import torch.nn as nn
from abc import ABCMeta


class PommerModel(nn.Module):
    def __init__(self, is_stateful):
        """
        Create a PommerModel.

        :param is_stateful: Whether the model is stateful (additionally receives and produces a state)
        """
        super().__init__()
        self.is_stateful = is_stateful

    def get_init_state(self, batch_size: int, device):
        """
        Get an initial state for the given batch size (only implemented if model is stateful)
        """
        pass
