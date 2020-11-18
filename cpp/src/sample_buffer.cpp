#include "sample_buffer.h"
#include "data_representation.h"

#include "xtensor/xarray.hpp"
#include "xtensor/xadapt.hpp"

SampleBuffer::SampleBuffer(const unsigned long capacity) : capacity(capacity), count(0) {
    this->obs = new float[GetObsSize(this->capacity)];
    this->act = new int8_t[this->capacity];
    this->pol = new float[this->capacity * NUM_MOVES];
    this->q = new float[this->capacity * NUM_MOVES];
    this->val = new float[this->capacity];
}

SampleBuffer::~SampleBuffer() {
    delete[] this->obs;
    delete[] this->act;
    delete[] this->pol;
    delete[] this->q;
    delete[] this->val;
}

ulong SampleBuffer::addSamples(const SampleBuffer& otherBuffer, const ulong offset, const ulong n) {
    ulong numSamples = std::min(n, this->capacity - this->count);

    std::copy_n(otherBuffer.obs + GetObsSize(offset), GetObsSize(numSamples), this->obs + GetObsSize(this->count));
    std::copy_n(otherBuffer.act + offset, numSamples, this->act + this->count);
    std::copy_n(otherBuffer.pol + NUM_MOVES * offset, NUM_MOVES * numSamples, this->pol + NUM_MOVES * this->count);
    std::copy_n(otherBuffer.q + NUM_MOVES * offset, NUM_MOVES * numSamples, this->q + NUM_MOVES * this->count);
    std::copy_n(otherBuffer.val + offset, numSamples, this->val + this->count);

    this->count += numSamples;

    return numSamples;
}

bool SampleBuffer::addSample(const float* planes, const bboard::Move move, const float moveProbs[NUM_MOVES], const float val, const float q[NUM_MOVES]) {
    if(count >= capacity)
        return false;

    std::copy_n(planes, GetObsSize(1), this->obs + GetObsSize(this->count));
    std::copy_n(moveProbs, NUM_MOVES, this->pol + NUM_MOVES * this->count);
    std::copy_n(q, NUM_MOVES, this->q + NUM_MOVES * this->count);
    act[count] = int8_t(move);
    this->val[count] = val;
    count += 1;

    return true;
}

bool SampleBuffer::addSample(const float* planes, const bboard::Move move) {
    float moveProbs[NUM_MOVES];
    std::fill_n(moveProbs, NUM_MOVES, 0);
    const int m = (int)move;
    if(m >= 0 && m <= NUM_MOVES - 1)
    {
        moveProbs[m] = 1.0f;
    }

    float q[NUM_MOVES];
    std::fill_n(q, NUM_MOVES, std::numeric_limits<float>::quiet_NaN());

    return addSample(planes, move, moveProbs, 0, q);
}

void SampleBuffer::setValues(const float value) {
    std::fill_n(val, count, value);
}

void SampleBuffer::setValuesDiscounted(const float value, const float discountFactor, bool addWeightedValues) {
    if (count <= 0)
        return;

    // set the final value
    val[count - 1] = value;

    // discount the others
    if (addWeightedValues) {
        float runningDiscountFactor = discountFactor;
        for (int i = count - 2; i >= 0;--i) {
            val[i] = runningDiscountFactor * value + (1 - runningDiscountFactor) * val[i];
            runningDiscountFactor *= discountFactor;
        }
    }
    else {
        float runningValue = discountFactor * value;
        for (int i = count - 2; i >= 0;--i) {
            val[i] = runningValue;
            runningValue *= discountFactor;
        }
    }
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

const float* SampleBuffer::getPol() const {
    return this->pol;
}

const float* SampleBuffer::getVal() const {
    return this->val;
}

const float* SampleBuffer::getQ() const {
    return this->q;
}

ulong SampleBuffer::getCount() const {
    return this->count;
}

ulong SampleBuffer::getCapacity() const {
    return this->capacity;
}

ulong SampleBuffer::getTotalObsValCount() const {
    return GetObsSize(this->getCount());
}
