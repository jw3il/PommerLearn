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
    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, sample: PommerSample):
        idx = np.random.randint(0, len(self.transforms))
        transform = self.transforms[idx]
        return transform(sample)


class ComposeTransform:
    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, sample: PommerSample):
        for transform in self.transforms:
            sample = transform(sample)

        return sample


class Identity:
    """
    Returns the identity
    """

    def __call__(self, sample: PommerSample):
        return sample


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
