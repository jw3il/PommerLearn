from data_augmentation import FlipX, FlipY, Rotate90
from dataset_util import PommerDataset

data = PommerDataset.from_zarr('mcts_data_500.zr', 0.9)

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
print_obs(transform(sample_0), transform(sample_1))
