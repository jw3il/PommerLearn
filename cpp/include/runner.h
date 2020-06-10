#ifndef RUNNER_H
#define RUNNER_H

#include "bboard.hpp"
#include "ipc_manager.h"
#include "episode_info.h"

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
     * @param maxEpisodeSteps The maximum number of steps per episode.
     * @param maxEpisodes The maximum number of episodes. Ignored if -1.
     * @param maxTotalSteps The (minimum) total number of simulated steps. Starts new episodes when this limit is not reached. Ignored if -1.
     */
    void generateSupervisedTrainingData(IPCManager* ipcManager, int maxEpisodeSteps, int maxEpisodes, int maxTotalSteps);

private:

};

#endif // RUNNER_H
