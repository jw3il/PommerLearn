#ifndef SAMPLE_BUFFER_H
#define SAMPLE_BUFFER_H

#include "bboard.hpp"

#include <cstdint>

/**
 * @brief The SampleBuffer class allows to convert and buffer generated samples.
 */
class SampleBuffer {
public:
    SampleBuffer(const ulong capacity);

    /**
     * @brief addSamples Converts samples and adds them to the buffer, as long as the capacity is not reached.
     * @param states Pointer to an array which contains at least "count" states.
     * @param moves Pointer to an array which contains at least "count" moves.
     * @param value The value associated with all samples.
     * @param agentId The id of the agent which collected these samples.
     * @param count The number of samples.
     * @return The number of added samples (is <= count, depending on the capacity of the buffer).
     */
    ulong addSamples(const bboard::State* states, const bboard::Move* moves, const float value, const int agentId, const ulong count);

    /**
     * @brief clear Clears the buffer state (not its actual content).
     */
    void clear();

    const float* getObs() const;
    const int8_t* getAct() const;
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
    float* val;
};

#endif // SAMPLE_BUFFER_H
