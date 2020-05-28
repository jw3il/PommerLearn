#ifndef LOGAGENT_H
#define LOGAGENT_H

#include "bboard.hpp"

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

    /**
     * @brief stateBuffer Contains the states of the current/last episode.
     */
    bboard::State* stateBuffer;

    /**
     * @brief actionBuffer Contains the actions of the current/last episode.
     */
    bboard::Move* actionBuffer;

    /**
     * @brief step The number of steps of the current/last episode.
     */
    int step;

    /**
     * @brief won Whether this agent has won its last episode.
     */
    bool won;

    bboard::Move act(const bboard::State* state) override;

    /**
     * @brief reset Reset the agent with a new controlling agent.
     * @param agent The agent which decides which actions this LogAgent will use.
     */
    void reset(bboard::Agent* agent);

    /**
     * @brief deleteAgent Deletes the agent if it exists.
     */
    void deleteAgent();

private:
    bboard::Agent* agent;
};

#endif // LOGAGENT_H
