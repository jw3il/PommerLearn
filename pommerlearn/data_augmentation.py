import math
from typing import NamedTuple, Tuple, List

import numpy as np
import torch
from pommerman.constants import Action


class PommerSample(NamedTuple):
    obs: torch.Tensor
    val: torch.Tensor
    act: torch.Tensor
    pol: torch.Tensor


class RandomTransform:
    """
    Randomly selects one of the given transforms.
    """

    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, sample: PommerSample):
        idx = np.random.randint(0, len(self.transforms))
        transform = self.transforms[idx]
        return transform(sample)


class ComposeTransform:
    """
    Composes multiple transforms (nested transform).
    """

    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, sample: PommerSample):
        for transform in self.transforms:
            sample = transform(sample)

        return sample


class Identity:
    """
    Does not modify the sample.
    """

    def __call__(self, sample: PommerSample):
        return sample


class Rotate90:
    """
    Rotates a pommerman observation counter-clockwise by 90 degrees (multiple times).
    """

    def __init__(self, k):
        """
        Constructor

        :param k: Rotate the observation k times counter-clockwise by 90 degrees.
        """
        self.k = k
        self.action_permutation, self.action_permutation_inv = self._get_action_permutation(self.k)

    @staticmethod
    def _get_action_permutation(k) -> Tuple[List[int], List[int]]:
        """
        Gets the forward and inverse action permutation for rotating the actions k times counter-clockwise.

        Forward permutation a: Allows you to rotate indices like rotated_action_values = action_values[a]
        Inverse permutation b: range(6)[a][b] = range(6)[b][a] = range(6)

        :param k: The number of times to rotate the actions.
        :returns: Tuple for action permutations and inverse action permutations.
        """

        # actions are: Stop, Up, Down, Left, Right, Bomb
        # => 0 and 5 are fixed, 1-4 have to change according to the rotation
        act = np.arange(6)
        act_inverse = np.arange(6)

        # rotate left
        #   U          R
        # L   R  =>  U   D
        #   D          L
        # Up <- Right, Down <- Left, Left <- Up, Right <- Down
        idx_from = [1, 2, 3, 4]
        idx_to = [4, 3, 1, 2]

        for a in range(0, k):
            act[idx_from] = act[idx_to]
            # inverse directions
            act_inverse[idx_to] = act_inverse[idx_from]

        return list(act), list(act_inverse)

    def __call__(self, sample: PommerSample):
        # Rotate observation
        obs_new = sample.obs.rot90(self.k, dims=(1, 2))

        # Rotate action
        act_new = sample.act.clone()
        for a in [1, 2, 3, 4]:
            # as action values == action indices, we can use the inverse permutation to get the rotated action values
            # (the inverse will always point to the indices our actions move in the forward permutation,
            # i.e. forward[left] = up and inverse[up] = left for k = 1)
            act_new[sample.act == a] = self.action_permutation_inv[a]

        # Rotate policy using the (forward) permutation
        pol_new = sample.pol.clone()[self.action_permutation]

        val_new = sample.val.clone()

        return PommerSample(obs_new, val_new, act_new, pol_new)


class FlipX:
    """
    Flips a pommerman observation along the x axis.
    """

    def __call__(self, sample: PommerSample):
        # Flip observation along x-axis (plane, y, x)
        obs_new = sample.obs.flip(dims=[-1])

        # Switch left & right actions
        act_left_filter = (sample.act == Action.Left.value)
        act_right_filter = (sample.act == Action.Right.value)

        act_new = sample.act.clone()
        act_new[act_left_filter] = Action.Right.value
        act_new[act_right_filter] = Action.Left.value

        # Switch left & right policy
        pol_left = sample.pol[Action.Left.value]
        pol_right = sample.pol[Action.Right.value]

        pol_new = sample.pol.clone()
        pol_new[Action.Left.value] = pol_right
        pol_new[Action.Right.value] = pol_left

        val_new = sample.val.clone()

        return PommerSample(obs_new, val_new, act_new, pol_new)


class FlipY:
    """
    Flips a pommerman observation along the y axis.
    """

    def __call__(self, sample: PommerSample):
        # Flip observation along y-axis (plane, y, x)
        obs_new = sample.obs.flip(dims=[-2])

        # Switch up & down actions
        act_up_filter = (sample.act == Action.Up.value)
        act_down_filter = (sample.act == Action.Down.value)

        act_new = sample.act.clone()
        act_new[act_up_filter] = Action.Down.value
        act_new[act_down_filter] = Action.Up.value

        # Switch up & down policy
        pol_up = sample.pol[Action.Up.value]
        pol_down = sample.pol[Action.Down.value]

        pol_new = sample.pol.clone()
        pol_new[Action.Up.value] = pol_down
        pol_new[Action.Down.value] = pol_up

        val_new = sample.val.clone()

        return PommerSample(obs_new, val_new, act_new, pol_new)
