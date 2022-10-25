#ifndef DATA_REPRESENTATION_H
#define DATA_REPRESENTATION_H

#include "bboard.hpp"
#include "xtensor/xarray.hpp"

const int PLANE_COUNT = 18;
const int PLANE_SIZE = bboard::BOARD_SIZE;
const int PLANES_TOTAL_FLOATS = PLANE_COUNT * PLANE_SIZE * PLANE_SIZE;

const int NUM_MOVES = 6;
extern bool centeredView; // option for centering the observation around the agent

inline long GetObsSize(const long step) {
    return step * PLANES_TOTAL_FLOATS;
}

/*
Action Space

0 = Stop: This action is a pass.
1 = Up: Move up on the board.
2 = Down: Move left on the board.
3 = Left: Move down on the board.
4 = Right: Move right on the board.
5 = Bomb: Lay a bomb.

See bboard::Move
*/

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

/**
 * @brief Converts the given board to input planes from the perspective of the given player id. Directly saves these planes in the given float array.
 * 
 * @param board The board.
 * @param id The id of the player.
 * @param planes A float pointer to a buffer of size PLANE_COUNT * PLANE_SIZE * PLANE_SIZE.
 * @param centeredView Agent is kept in the middle of the View. Limits the agents view to (board-size-1)/2 tiles.
 */
void BoardToPlanes(const bboard::Board* board, int id, float* planes);

/**
 * @brief InitialStateString Converts an initial state to a string representation. Warning: Has to be the initial state, does not handle bombs or flames.
 * @param state An initial state of the board.
 * @return A string which represents the given state.
 */
std::string InitialStateToString(const bboard::State& state);

#endif // DATA_REPRESENTATION_H
