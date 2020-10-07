#ifndef SAMPLE_BUFFER_H
#define SAMPLE_BUFFER_H

#include "bboard.hpp"
#include <cstdint>
#include "data_representation.h"

typedef unsigned long ulong;

/**
 * @brief The SampleBuffer class allows to convert and buffer generated samples.
 */
class SampleBuffer {
public:
    SampleBuffer(const ulong capacity);

    /**
     * @brief addSamples Copies samples from the given buffer until the capacity is reached.
     * @param buffer Content from otherBuffer[offset] to otherBuffer[otherBuffer.count - 1] is copied to this buffer.
     * @param offset The offset in the other buffer.
     * @param n The number of elements we want to copy (should be <= otherBuffer.count - offset)
     * @return The number of added samples (is <= n, depending on the capacity of this buffer).
     */
    ulong addSamples(const SampleBuffer& otherBuffer, const ulong offset, const ulong n);

    /**
     * @brief addSample Adds a single sample to the buffer, as long as the capacity is not reached.
     * @param planes Pointer to the input planes (observation).
     * @param moves The move which was chosen based on the given planes.
     * @return Whether the sample has been added (false if the capacity has already been reached).
     */
    bool addSample(const float* planes, const bboard::Move move);

    /**
     * @brief addSample Adds a single sample to the buffer, as long as the capacity is not reached.
     * @param planes Pointer to the input planes (observation).
     * @param moves The move which was chosen based on the given planes.
     * @param moveProbs The move probabilities
     * @return Whether the sample has been added (false if the capacity has already been reached).
     */
    bool addSample(const float* planes, const bboard::Move move, const float moveProbs[NUM_MOVES]);

    /**
     * @brief setValues Sets all values of the buffer according to the given value.
     * @param value Used to set val[0:count-1] = value.
     */
    void setValues(const float value);

    /**
     * @brief clear Clears the buffer state (not its actual content).
     */
    void clear();

    const float* getObs() const;
    const int8_t* getAct() const;
    const float* getPol() const;
    const float* getVal() const;

    ulong getCount() const;
    ulong getCapacity() const;
    ulong getTotalObsValCount() const;

    ~SampleBuffer();

private:
    ulong capacity;
    ulong count;

    float* obs;
    int8_t* act;
    float* pol;
    float* val;
};

#endif // SAMPLE_BUFFER_H
