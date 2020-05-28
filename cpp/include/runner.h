#ifndef RUNNER_H
#define RUNNER_H

#include "bboard.hpp"
#include "ipc_manager.h"

/**
 * @brief Contains metadata about a single episode.
 */
struct EpisodeInfo {
    int winner;
    int steps;
};

/**
 * @brief The Runner class povides utilities for running simulations in the pommerman environment.
 */
class Runner
{
public:
    Runner();
    /**
     * @brief run Simulate a single episode with the given agents and maximum steps.
     * @param agents The agents used in the episode.
     * @param maxSteps The maximum number of steps of the episode.
     * @return The result of the episode.
     */
    EpisodeInfo run(std::array<bboard::Agent*, 4> agents, int maxSteps);

    /**
     * @brief generateSupervisedTrainingData Generate a traning dataset for supervised training.
     * @param ipcManager The IPCManager which is used to save/transmit the episode logs.
     * @param maxSteps The maximum number of steps per episode.
     * @param episodes The number of episodes.
     */
    void generateSupervisedTrainingData(IPCManager* ipcManager, int maxSteps, int episodes);

private:

};

#endif // RUNNER_H
