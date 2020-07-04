#include "sample_buffer.h"
#include "data_representation.h"

#include "xtensor/xarray.hpp"
#include "xtensor/xadapt.hpp"

inline long _getObsSize(const long step) {
    return step * PLANE_COUNT * PLANE_SIZE * PLANE_SIZE;
}

SampleBuffer::SampleBuffer(const unsigned long capacity) : capacity(capacity), count(0)
{
    this->obs = new float[_getObsSize(this->capacity)];
    this->act = new int8_t[this->capacity];
    this->val = new float[this->capacity];
}

SampleBuffer::~SampleBuffer() {
    delete[] this->obs;
    delete[] this->act;
    delete[] this->val;
}

ulong SampleBuffer::addSamples(const bboard::State* states, const bboard::Move* moves, const float value, const int agentId, const ulong count)
{
    // skip samples which do not fit in the buffer
    ulong steps = std::min(count, this->capacity - this->count);

    // observations

    for (uint i = 0; i < steps; i++) {
        float* obsPointer = &this->obs[_getObsSize(this->count + i)];
        StateToPlanes(&states[i], agentId, obsPointer);
    }

    // actions

    std::vector<size_t> act_shape = { steps };
    auto xtActionBuffer = xt::adapt(moves, steps, xt::no_ownership(), act_shape);
    auto xtAct = xt::adapt(&this->act[this->count], steps, xt::no_ownership(), act_shape);
    xtAct = xt::cast<uint8_t>(xtActionBuffer);

    // values

    std::fill_n(&this->val[this->count], steps, value);

    this->count += steps;
    return steps;
}

void SampleBuffer::clear() {
    this->count = 0;
}

const float* SampleBuffer::getObs() const {
    return this->obs;
}

const int8_t* SampleBuffer::getAct() const {
    return this->act;
}

const float* SampleBuffer::getVal() const {
    return this->val;
}

ulong SampleBuffer::getCount() const {
    return this->count;
}

ulong SampleBuffer::getCapacity() const {
    return this->capacity;
}

ulong SampleBuffer::getTotalObsValCount() const {
    return _getObsSize(this->getCount());
}
