#ifndef LOGAGENT_H
#define LOGAGENT_H

#include "bboard.hpp"
#include "sample_buffer.h"

/**
 * @brief Logs the observations and actions of the underlying agent.
 */
struct LogAgent : bboard::Agent
{
public:
    /**
     * @brief LogAgent Create a LogAgent for logging episodes with the given maximum episode length.
     * @param maxEpisodeLength The maximum amount of steps per episode.
     */
    LogAgent(int maxEpisodeLength);
    ~LogAgent();

    /**
     * @brief sampleBuffer The samplebuffer used to save the experience of this agent.
     */
    SampleBuffer sampleBuffer;

    /**
     * @brief step The number of steps of the current/last episode.
     */
    uint step;

    bboard::Move act(const bboard::State* state) override;

    /**
     * @brief reset Reset the agent with a new controlling agent. Has to be called after Environment::MakeGame so that the id can be set correctly.
     * @param agent The agent which decides which actions this LogAgent will use.
     */
    void reset(bboard::Agent* agent);

    /**
     * @brief deleteAgent Deletes the agent if it exists.
     */
    void deleteAgent();

private:
    bboard::Agent* agent;
    float* planeBuffer;
};

#endif // LOGAGENT_H
