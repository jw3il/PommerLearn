#ifndef DATA_REPRESENTATION_H
#define DATA_REPRESENTATION_H

#include "bboard.hpp"
#include "xtensor/xarray.hpp"

const int PLANE_COUNT = 18;
const int PLANE_SIZE = bboard::BOARD_SIZE;
const int PLANES_TOTAL_FLOATS = PLANE_COUNT * PLANE_SIZE * PLANE_SIZE;

/*
Action Space

0 = Stop: This action is a pass.
1 = Up: Move up on the board.
2 = Left: Move left on the board.
3 = Down: Move down on the board.
4 = Right: Move right on the board.
5 = Bomb: Lay a bomb.

*/

/**
 * @brief MoveToInt Converts the given move to an integer.
 * @param move The move.
 * @return An int value which represents the given move.
 */
int8_t MoveToInt(bboard::Move move);

/*
Observation Space

Planes

Obstacles
* Non-Destructible
* Destructible

Items
* Increase Bomb Count
* Increase Bomb Strength (Range)
* Kick

Bomb:
* Bomb Position & Life 0 -> 1
* Bomb Blast Strength

Flames:
* Bomb Flame Position & Life 1 -> 0

Player
* Position Self
* Position Enemy 1
* Position Enemy 2
* Position Enemy 3

Scalar Feature Planes:
* Self: Player Bomb Strength
* Self: Bomb Count (Ammo)
* Self: Max Momb Count
* Self: Can Kick

*/

// TODO: Change from all planes to single plane array (for the current step/state) -> maybe with xt::xview
/**
 * @brief StateToPlanes Converts the given state variable to input planes from the perspective of the given player id. Directly saves these planes in the given xarray.
 * @param state The state of the board (including the current step in the episode used for the plane index).
 * @param id The id of the player.
 * @param allInputPlanes The input planes of shape (DATASET_SIZE, PLANE_COUNT, PLANE_SIZE, PLANE_SIZE).
 * @param inputIndex The input index at which the current state should be saved (0 <= inputIndex < DATASET_SIZE)
 */
void StateToPlanes(bboard::State state, int id, xt::xarray<float>& allInputPlanes, uint inputIndex);

#endif // DATA_REPRESENTATION_H
