#include "data_representation.h"

#include "xtensor/xview.hpp"

int8_t MoveToInt(bboard::Move move) {
    switch (move) {
        case bboard::Move::IDLE: return 0;
        case bboard::Move::UP: return 1;
        case bboard::Move::LEFT: return 2;
        case bboard::Move::DOWN: return 3;
        case bboard::Move::RIGHT: return 4;
        case bboard::Move::BOMB: return 5;
        default: return -1;
    }
}

float inline _getNormalizedBombStrength(int stength) {
    float val = (float)stength / bboard::BOARD_SIZE;
    return val > 1.0f ? 1.0f : val;
}

void StateToPlanes(bboard::State state, int id, xt::xarray<float>& allInputPlanes) {
    std::vector<std::size_t> planeShape = { PLANE_SIZE, PLANE_SIZE };

    // TOOD: use xt::adapt instead
    xt::xarray<int> board(planeShape);
    for (int y = 0; y < PLANE_SIZE; y++){
        for (int x = 0; x < PLANE_SIZE; x++){
            board(y, x) = state.board[y][x];
        }
    }

    int planeIndex = 0;

    // obstacle planes
    xt::view(allInputPlanes, state.timeStep, planeIndex++, xt::all(), xt::all()) = xt::cast<float>(xt::equal(board, bboard::Item::RIGID));
    xt::view(allInputPlanes, state.timeStep, planeIndex++, xt::all(), xt::all()) = xt::cast<float>(xt::equal(board, bboard::Item::WOOD));

    // item planes
    xt::view(allInputPlanes, state.timeStep, planeIndex++, xt::all(), xt::all()) = xt::cast<float>(xt::equal(board, bboard::Item::EXTRABOMB));
    xt::view(allInputPlanes, state.timeStep, planeIndex++, xt::all(), xt::all()) = xt::cast<float>(xt::equal(board, bboard::Item::INCRRANGE));
    xt::view(allInputPlanes, state.timeStep, planeIndex++, xt::all(), xt::all()) = xt::cast<float>(xt::equal(board, bboard::Item::KICK));


    // bomb planes (lifetime & strength)
    int bombTimePlane = planeIndex++;
    int bombStrengthPlane = planeIndex++;
    xt::view(allInputPlanes, state.timeStep, bombTimePlane, xt::all(), xt::all()) = 0;
    xt::view(allInputPlanes, state.timeStep, bombStrengthPlane, xt::all(), xt::all()) = 0;

    for (int i = 0; i < state.bombs.count; i++) {
        bboard::Bomb bomb = state.bombs[i];

        // bombs explode at BMB_TIME == 0, we invert that to get values from 0->1 until the bombs explode
        xt::view(allInputPlanes, state.timeStep, bombTimePlane, bboard::BMB_POS_Y(bomb), bboard::BMB_POS_X(bomb)) = 1 - ((float)bboard::BMB_TIME(bomb) / bboard::BOMB_LIFETIME);
        xt::view(allInputPlanes, state.timeStep, bombStrengthPlane, bboard::BMB_POS_Y(bomb), bboard::BMB_POS_X(bomb)) = _getNormalizedBombStrength(bboard::BMB_STRENGTH(bomb));
    }

    // flame plane (lifetime)
    int flamesPlane = planeIndex++;
    xt::view(allInputPlanes, state.timeStep, flamesPlane, xt::all(), xt::all()) = 0;

    for (int i = 0; i < state.flames.count; i++) {
        bboard::Flame flame = state.flames[i];

        // each flame object has an id which is used to identify the corresponding flame items on the board
        int flameId = bboard::FLAME_ID(board(flame.position.y, flame.position.x));
        float flameValue = (float)flame.timeLeft / bboard::FLAME_LIFETIME;

        // TODO: change manual loops to xtensor operations

        // a flame object represents an explosion which consists of multiple flame items on the board

        // row
        for (int x = 0; x < bboard::BOARD_SIZE; x++) {
            int bval = board(flame.position.y, x);
            if (bboard::IS_FLAME(bval) && bboard::FLAME_ID(bval) == flameId) {
                xt::view(allInputPlanes, state.timeStep, flamesPlane, flame.position.y, x) = flameValue;
            }
        }
        // column
        for (int y = 0; y < bboard::BOARD_SIZE; y++) {
            int bval = board(y, flame.position.x);
            if (bboard::IS_FLAME(bval) && bboard::FLAME_ID(bval) == flameId) {
                xt::view(allInputPlanes, state.timeStep, flamesPlane, y, flame.position.x) = flameValue;
            }
        }
    }

    // player position planes
    for (int i = 0; i < 4; i++) {
        int currentId = (id + i) % 4;

        // set plane to zero
        int currentPlane = planeIndex++;
        xt::view(allInputPlanes, state.timeStep, currentPlane, xt::all(), xt::all()) = 0;

        // only insert the position if the agent is alive
        bboard::AgentInfo agentInfo = state.agents[currentId];
        if (!agentInfo.dead) {
            xt::view(allInputPlanes, state.timeStep, currentPlane, agentInfo.y, agentInfo.x) = 1;
        }
    }

    // scalar feature planes
    bboard::AgentInfo info = state.agents[id];

    xt::view(allInputPlanes, state.timeStep, planeIndex++, xt::all(), xt::all()) = _getNormalizedBombStrength(info.bombStrength);
    xt::view(allInputPlanes, state.timeStep, planeIndex++, xt::all(), xt::all()) = (float)info.bombCount / bboard::MAX_BOMBS_PER_AGENT;
    xt::view(allInputPlanes, state.timeStep, planeIndex++, xt::all(), xt::all()) = (float)info.maxBombCount / bboard::MAX_BOMBS_PER_AGENT;
    xt::view(allInputPlanes, state.timeStep, planeIndex++, xt::all(), xt::all()) = info.canKick ? 1 : 0;
}
