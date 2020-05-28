#ifndef RUNNER_H
#define RUNNER_H

#include "bboard.hpp"
#include "ipc_manager.h"

/**
 * @brief Contains metadata about a single episode.
 */
struct EpisodeInfo {
    int winner;
    bool isDraw;
    bool isDone;
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
     * @brief run Simulate a single episode with the given environment and maximum steps.
     * @param env The environment to use. Must be initialized.
     * @param maxSteps The maximum number of steps of the episode.
     * @return The result of the episode.
     */
    EpisodeInfo run(bboard::Environment& env, int maxSteps);

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
