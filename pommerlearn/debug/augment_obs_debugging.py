import numpy as np
import torch

from data_augmentation import FlipX, FlipY, Rotate90
from dataset_util import PommerDataset, PommerSample

data = PommerDataset.from_zarr_path('mcts_data_500.zr', 0.9)

# search for bomb movement in observations (planes 7 & 8)
horizontal_bomb_movement = data.obs[:, 7].nonzero()
vertical_bomb_movement = data.obs[:, 8].nonzero()

# Manually: select an index where bomb movement is present and you can actually see movement between steps
index = horizontal_bomb_movement[0][0]
h_index = 7
v_index = 8
sample_0 = data[index]
sample_1 = data[index + 1]


def print_obs(s0, s1):
    print("Horizontal")
    print(s0.obs[h_index])
    print(s1.obs[h_index])
    print("Vertical")
    print(s0.obs[v_index])
    print(s1.obs[v_index])


print("Before transform")
print_obs(sample_0, sample_1)

print("After transform")
transform = Rotate90(3)
transformed_0 = transform(sample_0)
transformed_1 = transform(sample_1)

# check that batch transform returns the same results
both_transformed = transform(PommerSample.merge(sample_0, sample_1))
assert transformed_0.equals(both_transformed.batch_at(0))
assert transformed_1.equals(both_transformed.batch_at(1))

print_obs(transformed_0, transformed_1)
