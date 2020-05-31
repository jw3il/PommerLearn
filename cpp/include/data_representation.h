#ifndef DATA_REPRESENTATION_H
#define DATA_REPRESENTATION_H

#include "bboard.hpp"
#include "agents.hpp"

const int PLANE_COUNT = 11;
const int PLANE_WIDTH = 11;
const int PLANE_HEIGHT = 11;
const int PLANE_SIZE = PLANE_WIDTH * PLANE_HEIGHT;

const int STATE_FLOAT_COUNT = PLANE_COUNT * PLANE_SIZE;

int MoveToInt(bboard::Move move);

void StateToPlanes(bboard::State state, float* inputPlanes);

#endif // DATA_REPRESENTATION_H
