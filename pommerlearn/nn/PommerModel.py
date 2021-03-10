import abc
from typing import Optional, Tuple

import torch
import torch.nn as nn
from abc import ABCMeta


class PommerModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, is_stateful, state_batch_dim):
        """
        Create a PommerModel.

        :param is_stateful: Whether the model is stateful (additionally receives and produces a state)
        :param state_batch_dim: The batch dimension of the state (required for flattening if model is stateful)
        """
        super().__init__()
        self.is_stateful = is_stateful
        self.state_batch_dim = state_batch_dim

        self.sequence_length = None
        self.has_state_input = None

    def get_init_state_bf(self, batch_size: int, device):
        """
        Get an initial state for the given batch size with the batch dimension first
        (only required if the model is stateful).

        :param batch_size: The batch size
        :returns: The shape of a state, as required by the used stateful module
        """
        bf_shape = self.transpose_state_shape(self.get_state_shape(batch_size))
        return torch.zeros(bf_shape, requires_grad=False).to(device)

    @abc.abstractmethod
    def get_state_shape(self, batch_size: int) -> Tuple[int]:
        """
        Gets the (original) shape of a state (only required if the model is stateful).

        :param batch_size: The batch size
        :returns: The shape of a state, as required by the used stateful module
        """
        ...

    def set_input_options(self, sequence_length: Optional[int], has_state_input: bool):
        self.sequence_length = sequence_length
        self.has_state_input = has_state_input

    def transpose_state(self, state):
        """
        A transpose operation which transforms a batch_first (bf) state to the dims expected by the used stateful
        Module and vice versa.
        """
        # just switch state and batch dim
        return torch.transpose(state, self.state_batch_dim, 0).contiguous()

    def transpose_state_shape(self, state_shape):
        transposed_shape = list(state_shape)
        transposed_shape[self.state_batch_dim] = state_shape[0]
        transposed_shape[0] = state_shape[self.state_batch_dim]

        return transposed_shape

    def flatten(self, x: torch.Tensor, state_bf: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Flattens and concatenates model input and state tensors.

        :param x: The (spacial) input of the model.
        :param state_bf: The current state (batch-dim first).
        :returns: A flattened concatenation of the given tensors
        """
        batch_size = x.shape[0]

        if state_bf is None:
            return x.view(batch_size, -1)

        # state is not none, we have to append the state to each sequence in the batch
        x_batches = x.view(batch_size, -1)
        state_bf_batches = state_bf.view(batch_size, -1)

        # concatenate the batches
        return torch.cat((x_batches, state_bf_batches), dim=1)
