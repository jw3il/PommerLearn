#include "data_representation.h"

#include "xtensor/xview.hpp"
#include "xtensor/xadapt.hpp"
#include <sstream>

float inline _getNormalizedBombStrength(int stength)
{
    float val = (float)stength / bboard::BOARD_SIZE;
    return val > 1.0f ? 1.0f : val;
}

template <typename xtPlanesType>
inline void _boardToPlanes(const bboard::Board* board, int id, xtPlanesType xtPlanes, int& planeIndex)
{
    // reset content
    xt::view(xtPlanes, xt::all()) = 0;

    // obstacle planes
    int rigidPlane = planeIndex++;
    int woodPlane = planeIndex++;

    // item planes
    int extraBombPlane = planeIndex++;
    int incrangePlane = planeIndex++;
    int kickPlane = planeIndex++;

    // bomb planes
    int bombTimePlane = planeIndex++;
    int bombStrengthPlane = planeIndex++;
    int bombMovementHorizontalPlane = planeIndex++;
    int bombMovementVerticalPlane = planeIndex++;

    int flamesPlane = planeIndex++;
    
    int agent0Plane = planeIndex++;
    int agent1Plane = planeIndex++;
    int agent2Plane = planeIndex++;
    int agent3Plane = planeIndex++;

    for (int y = 0; y < bboard::BOARD_SIZE; y++) {
        for (int x = 0; x < bboard::BOARD_SIZE; x++) {
            const bboard::Item item = static_cast<bboard::Item>(board->items[y][x]);
            if (bboard::IS_WOOD(item)) {
                xt::view(xtPlanes, woodPlane, y, x) = 1;
                continue;
            }
            switch (item)
            {
            case bboard::Item::RIGID:
            {
                xt::view(xtPlanes, rigidPlane, y, x) = 1;
                break;
            }
            case bboard::Item::EXTRABOMB:
            {
                xt::view(xtPlanes, extraBombPlane, y, x) = 1;
                break;
            }
            case bboard::Item::INCRRANGE:
            {
                xt::view(xtPlanes, incrangePlane, y, x) = 1;
                break;
            }
            case bboard::Item::KICK:
            {
                xt::view(xtPlanes, kickPlane, y, x) = 1;
                break;
            }
            case bboard::Item::AGENT0:
            {
                xt::view(xtPlanes, agent0Plane, y, x) = 1;
                break;
            }
            case bboard::Item::AGENT1:
            {
                xt::view(xtPlanes, agent1Plane, y, x) = 1;
                break;
            }
            case bboard::Item::AGENT2:
            {
                xt::view(xtPlanes, agent2Plane, y, x) = 1;
                break;
            }
            case bboard::Item::AGENT3:
            {
                xt::view(xtPlanes, agent3Plane, y, x) = 1;
                break;
            }
            default:
            {
                break;
            }
            }
        }
    }

    for (int i = 0; i < board->bombs.count; i++)
    {
        bboard::Bomb bomb = board->bombs[i];
        int x = bboard::BMB_POS_X(bomb);
        int y = bboard::BMB_POS_Y(bomb);

        // bombs explode at BMB_TIME == 0, we invert that to get values from 0->1 until the bombs explode
        xt::view(xtPlanes, bombTimePlane, y, x) = 1 - ((float)bboard::BMB_TIME(bomb) / bboard::BOMB_LIFETIME);
        xt::view(xtPlanes, bombStrengthPlane, y, x) = _getNormalizedBombStrength(bboard::BMB_STRENGTH(bomb));

        // bomb movement
        bboard::Move bombMovement = bboard::Move(bboard::BMB_DIR(bomb));
        switch (bombMovement)
        {
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
    float cumulativeTimeLeft = 0;
    for (int i = 0; i < board->flames.count; i++)
    {
        const bboard::Flame& flame = board->flames[i];

        cumulativeTimeLeft += (float)flame.timeLeft;
        float flameValue = cumulativeTimeLeft / bboard::FLAME_LIFETIME;
        xt::view(xtPlanes, flamesPlane, flame.position.y, flame.position.x) = flameValue;
    }
}

template <typename xtPlanesType>
inline void _infoToPlanes(const bboard::AgentInfo* info, xtPlanesType xtPlanes, int& planeIndex)
{
    xt::view(xtPlanes, planeIndex++) = _getNormalizedBombStrength(info->bombStrength);
    xt::view(xtPlanes, planeIndex++) = (float)info->bombCount / bboard::MAX_BOMBS_PER_AGENT;
    xt::view(xtPlanes, planeIndex++) = (float)info->maxBombCount / bboard::MAX_BOMBS_PER_AGENT;
    xt::view(xtPlanes, planeIndex++) = info->canKick ? 1 : 0;
}

void BoardToPlanes(const bboard::Board* board, int id, float* planes)
{
    // shape of all planes of a state
    std::vector<std::size_t> stateShape = { PLANE_COUNT, PLANE_SIZE, PLANE_SIZE };
    auto xtPlanes = xt::adapt(planes, PLANE_COUNT * PLANE_SIZE * PLANE_SIZE, xt::no_ownership(), stateShape);

    int planeIndex = 0;
    _boardToPlanes(board, id, xtPlanes, planeIndex);
    _infoToPlanes(&board->agents[id], xtPlanes, planeIndex);
}

std::string InitialStateToString(const bboard::State& state) {
    std::stringstream stream;

    for (int y = 0; y < bboard::BOARD_SIZE; y++) {
        for (int x = 0; x < bboard::BOARD_SIZE; x++) {
            int elem = state.items[y][x];

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
                                std::cerr << "Error: Encountered unknown item at (" << x << ", " << y << "): " << elem << ", item:" << item << std::endl;
                                // treat as regular wood
                                stream << "2";
                                break;
                        }
                    }
                    else {
                        std::cerr << "Error: Encountered unknown element at (" << x << ", " << y << "): " << elem << std::endl;
                        // ignore everything else
                        stream << "0";
                    }

                    break;
            }
        }
    }

    return stream.str();
}
