#include "data_representation.h"

#include "xtensor/xview.hpp"
#include "xtensor/xadapt.hpp"
#include <sstream>

float inline _getNormalizedBombStrength(int stength) {
    float val = (float)stength / bboard::BOARD_SIZE;
    return val > 1.0f ? 1.0f : val;
}

void StateToPlanes(const bboard::State* state, int id, float* planes) {
    // shape of a single plane
    std::vector<std::size_t> planeShape = { PLANE_SIZE, PLANE_SIZE };
    // adapt state.board without copying its values
    auto board = xt::adapt(&(state->board[0][0]), PLANE_SIZE * PLANE_SIZE, xt::no_ownership(), planeShape);

    // shape of all planes of a state
    std::vector<std::size_t> stateShape = { PLANE_COUNT, PLANE_SIZE, PLANE_SIZE };
    auto xtPlanes = xt::adapt(planes, PLANE_COUNT * PLANE_SIZE * PLANE_SIZE, xt::no_ownership(), stateShape);

    int planeIndex = 0;

    // obstacle planes
    xt::view(xtPlanes, planeIndex++) = xt::cast<float>(xt::equal(board, bboard::Item::RIGID));
    xt::view(xtPlanes, planeIndex++) = xt::cast<float>(xt::equal(board, bboard::Item::WOOD));

    // item planes
    xt::view(xtPlanes, planeIndex++) = xt::cast<float>(xt::equal(board, bboard::Item::EXTRABOMB));
    xt::view(xtPlanes, planeIndex++) = xt::cast<float>(xt::equal(board, bboard::Item::INCRRANGE));
    xt::view(xtPlanes, planeIndex++) = xt::cast<float>(xt::equal(board, bboard::Item::KICK));

    // bomb planes (lifetime & strength)
    int bombTimePlane = planeIndex++;
    int bombStrengthPlane = planeIndex++;
    int bombMovementHorizontalPlane = planeIndex++;
    int bombMovementVerticalPlane = planeIndex++;

    xt::view(xtPlanes, bombTimePlane) = 0;
    xt::view(xtPlanes, bombStrengthPlane) = 0;
    xt::view(xtPlanes, bombMovementHorizontalPlane) = 0;
    xt::view(xtPlanes, bombMovementVerticalPlane) = 0;

    for (int i = 0; i < state->bombs.count; i++) {
        bboard::Bomb bomb = state->bombs[i];
        int x = bboard::BMB_POS_X(bomb);
        int y = bboard::BMB_POS_Y(bomb);

        // bombs explode at BMB_TIME == 0, we invert that to get values from 0->1 until the bombs explode
        xt::view(xtPlanes, bombTimePlane, y, x) = 1 - ((float)bboard::BMB_TIME(bomb) / bboard::BOMB_LIFETIME);
        xt::view(xtPlanes, bombStrengthPlane, y, x) = _getNormalizedBombStrength(bboard::BMB_STRENGTH(bomb));

        // bomb movement
        bboard::Move bombMovement = bboard::Move(bboard::BMB_DIR(bomb));
        switch (bombMovement) {
            case bboard::Move::UP:
                xt::view(xtPlanes, bombMovementVerticalPlane, y, x) = 1.0f;
                break;
            case bboard::Move::DOWN:
                xt::view(xtPlanes, bombMovementVerticalPlane, y, x) = -1.0f;
                break;
            case bboard::Move::LEFT:
                xt::view(xtPlanes, bombMovementHorizontalPlane, y, x) = -1.0f;
                break;
            case bboard::Move::RIGHT:
                xt::view(xtPlanes, bombMovementHorizontalPlane, y, x) = 1.0f;
                break;

            default: break;
        }
    }

    // flame plane (lifetime)
    int flamesPlane = planeIndex++;
    xt::view(xtPlanes, flamesPlane) = 0;

    for (int i = 0; i < state->flames.count; i++) {
        bboard::Flame flame = state->flames[i];

        // each flame object has an id which is used to identify the corresponding flame items on the board
        int flameId = bboard::FLAME_ID(board(flame.position.y, flame.position.x));
        float flameValue = (float)flame.timeLeft / bboard::FLAME_LIFETIME;

        // TODO: change manual loops to xtensor operations

        // a flame object represents an explosion which consists of multiple flame items on the board

        // row
        for (int x = 0; x < bboard::BOARD_SIZE; x++) {
            int bval = board(flame.position.y, x);
            if (bboard::IS_FLAME(bval) && bboard::FLAME_ID(bval) == flameId) {
                xt::view(xtPlanes, flamesPlane, flame.position.y, x) = flameValue;
            }
        }
        // column
        for (int y = 0; y < bboard::BOARD_SIZE; y++) {
            int bval = board(y, flame.position.x);
            if (bboard::IS_FLAME(bval) && bboard::FLAME_ID(bval) == flameId) {
                xt::view(xtPlanes, flamesPlane, y, flame.position.x) = flameValue;
            }
        }
    }

    // player position planes
    for (int i = 0; i < 4; i++) {
        int currentId = (id + i) % 4;

        // set plane to zero
        int currentPlane = planeIndex++;
        xt::view(xtPlanes, currentPlane) = 0;

        // only insert the position if the agent is alive
        bboard::AgentInfo agentInfo = state->agents[currentId];
        if (!agentInfo.dead) {
            xt::view(xtPlanes, currentPlane, agentInfo.y, agentInfo.x) = 1;
        }
    }

    // scalar feature planes
    bboard::AgentInfo info = state->agents[id];

    xt::view(xtPlanes, planeIndex++) = _getNormalizedBombStrength(info.bombStrength);
    xt::view(xtPlanes, planeIndex++) = (float)info.bombCount / bboard::MAX_BOMBS_PER_AGENT;
    xt::view(xtPlanes, planeIndex++) = (float)info.maxBombCount / bboard::MAX_BOMBS_PER_AGENT;
    xt::view(xtPlanes, planeIndex++) = info.canKick ? 1 : 0;
}

std::string InitialStateToString(bboard::State state) {
    std::stringstream stream;

    for (int y = 0; y < bboard::BOARD_SIZE; y++) {
        for (int x = 0; x < bboard::BOARD_SIZE; x++) {
            int elem = state.board[y][x];

            switch (elem) {
                case bboard::Item::PASSAGE: stream << "0"; break;
                case bboard::Item::RIGID:   stream << "1"; break;
                case bboard::Item::WOOD:    stream << "2"; break;
                case bboard::Item::AGENT0:  stream << "A"; break;
                case bboard::Item::AGENT1:  stream << "B"; break;
                case bboard::Item::AGENT2:  stream << "C"; break;
                case bboard::Item::AGENT3:  stream << "D"; break;
                default:
                    if (bboard::IS_WOOD(elem)) {
                        int item = state.FlagItem(bboard::WOOD_POWFLAG(elem));
                        switch (item) {
                            case bboard::EXTRABOMB: stream << "3"; break;
                            case bboard::INCRRANGE: stream << "4"; break;
                            case bboard::KICK:      stream << "5"; break;
                            // when we do not know this item, treat it like a regular wood (should not happen!)
                            default:
                                std::cerr << "Error: Encountered unknown item at (" << x << ", " << y << "): " << std::hex << elem << ", item:" << std::dec << item << std::endl;
                                // treat as regular wood
                                stream << "2";
                                break;
                        }
                    }
                    else {
                        std::cerr << "Error: Encountered unknown element at (" << x << ", " << y << "): " << std::hex << elem << std::endl;
                        // ignore everything else
                        stream << "0";
                    }

                    break;
            }
        }
    }

    return stream.str();
}
