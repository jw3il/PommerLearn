#include "data_representation.h"

int MoveToInt(bboard::Move move) {
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

void StateToPlanes(bboard::State state, float* inputPlanes) {
    std::fill(inputPlanes, inputPlanes + STATE_FLOAT_COUNT, 0.0f);
    // TODO: Convert state to planes
}
