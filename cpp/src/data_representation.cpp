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
    // shape of a single plane (BOARD_SIZE == PLANE_SIZE)
    std::vector<std::size_t> boardShape = { bboard::BOARD_SIZE, bboard::BOARD_SIZE };
    // adapt state.board without copying its values
    auto items = xt::adapt(&(board->items[0][0]), bboard::BOARD_SIZE * bboard::BOARD_SIZE, xt::no_ownership(), boardShape);

    // obstacle planes
    xt::view(xtPlanes, planeIndex++) = xt::cast<float>(xt::equal(items, bboard::Item::RIGID));
    // wood blocks can also contain an item (+0, +1, +2, +3)
    auto isWood = xt::equal(items, bboard::Item::WOOD) + xt::equal(items, bboard::Item::WOOD + 1)
            + xt::equal(items, bboard::Item::WOOD + 2) + xt::equal(items, bboard::Item::WOOD + 3);
    xt::view(xtPlanes, planeIndex++) = xt::cast<float>(isWood);

    // item planes
    xt::view(xtPlanes, planeIndex++) = xt::cast<float>(xt::equal(items, bboard::Item::EXTRABOMB));
    xt::view(xtPlanes, planeIndex++) = xt::cast<float>(xt::equal(items, bboard::Item::INCRRANGE));
    xt::view(xtPlanes, planeIndex++) = xt::cast<float>(xt::equal(items, bboard::Item::KICK));

    // bomb planes (lifetime & strength)
    int bombTimePlane = planeIndex++;
    int bombStrengthPlane = planeIndex++;
    int bombMovementHorizontalPlane = planeIndex++;
    int bombMovementVerticalPlane = planeIndex++;

    xt::view(xtPlanes, bombTimePlane) = 0;
    xt::view(xtPlanes, bombStrengthPlane) = 0;
    xt::view(xtPlanes, bombMovementHorizontalPlane) = 0;
    xt::view(xtPlanes, bombMovementVerticalPlane) = 0;

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
    int flamesPlane = planeIndex++;
    xt::view(xtPlanes, flamesPlane) = 0;

    float cumulativeTimeLeft = 0;
    for (int i = 0; i < board->flames.count; i++)
    {
        const bboard::Flame& flame = board->flames[i];

        cumulativeTimeLeft += (float)flame.timeLeft;
        float flameValue = cumulativeTimeLeft / bboard::FLAME_LIFETIME;
        xt::view(xtPlanes, flamesPlane, flame.position.y, flame.position.x) = flameValue;
    }

    // player position planes
    for (int i = 0; i < 4; i++)
    {
        int currentId = (id + i) % 4;

        int currentPlane = planeIndex++;
        xt::view(xtPlanes, currentPlane) = xt::cast<float>(xt::equal(items, bboard::Item::AGENT0 + currentId));;
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

void StateToPlanes(const bboard::State* state, int id, float* planes)
{
    // shape of all planes of a state
    std::vector<std::size_t> stateShape = { PLANE_COUNT, PLANE_SIZE, PLANE_SIZE };
    auto xtPlanes = xt::adapt(planes, PLANE_COUNT * PLANE_SIZE * PLANE_SIZE, xt::no_ownership(), stateShape);

    int planeIndex = 0;
    _boardToPlanes(state, id, xtPlanes, planeIndex);
    _infoToPlanes(&state->agents[id], xtPlanes, planeIndex);
}

void ObservationToPlanes(const bboard::Observation* obs, int id, float* planes)
{
    // shape of all planes of a state
    std::vector<std::size_t> stateShape = { PLANE_COUNT, PLANE_SIZE, PLANE_SIZE };
    auto xtPlanes = xt::adapt(planes, PLANE_COUNT * PLANE_SIZE * PLANE_SIZE, xt::no_ownership(), stateShape);

    int planeIndex = 0;
    _boardToPlanes(obs, id, xtPlanes, planeIndex);
    _infoToPlanes(&obs->agentInfos[obs->agentIDMapping[id]], xtPlanes, planeIndex);
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
