#ifndef RUNNER_H
#define RUNNER_H

#include "bboard.hpp"
#include "agents.hpp"
#include "ipc_manager.h"
#include "episode_info.h"
#include "pommermanstate.h"

/**
 * @brief Configuration struct for the runner class.
 */
struct RunnerConfig
{
    /**
     * @brief gameMode The game mode for all episodes.
     */
    bboard::GameMode gameMode;

    /**
     * @brief observationParameters The observation parameters for all agents in all episodes.
     */
    bboard::ObservationParameters observationParameters;

    /**
     * @brief useVirtualStep Parameter defining wetherto recreate information from previous observations
     */
    bool useVirtualStep = false;

    /**
     * @brief maxEpisodeSteps The maximum number of steps per episode.
     */
    int maxEpisodeSteps = 800;

    /**
     * @brief maxEpisodes The maximum number of episodes. Ignored if -1.
     */
    long maxEpisodes = 10;

    /**
     * @brief targetedLoggedSteps The targeted number of logged steps after this call (soft limit: still add complete episodes). Starts new episodes when this limit is not reached. Ignored if -1.
     */
    long targetedLoggedSteps = -1;

    /**
     * @brief maxLoggedSteps The maximum total number of logged steps after this call. Starts new episodes when this limit is not reached. Ignored if -1.
     */
    long maxLoggedSteps = -1;

    /**
     * @brief seed The seed used to generate the training data (for deterministic results). Ignored (initialized with time) if -1.
     */
    long seed = -1;

    /**
     * @brief envSeed The seed used to generate the environment. If -1, every episode will use a randomly generated environment.
     */
    long envSeed = -1;

    /**
     * @brief envGenSeedEps The number of episodes after which a new environment generation seed is created.
     */
    long envGenSeedEps = 0;

    /**
     * @brief randomAgentPositions Whether to use random agent positions in every episode.
     */
    bool randomAgentPositions = true;

    /**
     * @brief printSteps Whether to print the steps.
     */
    bool printSteps = false;

    /**
     * @brief printFirstLast Whether to print the first and last steps of each episode.
     */
    bool printFirstLast = false;

    /**
     * @brief useStateInSearch Whether to use the true environment state in CrazyAraAgents.
     */
    bool useStateInSearch = true;

    /**
     * @brief ipcManager The IPCManager which is used to save/transmit the episode logs. No logs are saved if this is a nullptr.
     */
    IPCManager* ipcManager = nullptr;

    /**
     * @brief flushEpisodes The number of episodes until the stored data is flushed by the runner. -1 for no flushing.
     */
    int flushEpisodes = -1;
};

/**
 * @brief The Runner class povides utilities for running simulations in the pommerman environment.
 */
class Runner
{
public:
    Runner();
    /**
     * @brief run_episode Simulate a single episode with the given environment and maximum steps.
     * @param env The environment to use. Must be initialized.
     * @param maxSteps The maximum number of steps of the episode.
     * @param printSteps Whether to print the steps.
     * @param printFirstLast Whether to print the first and last steps.
     * @return The result of the episode.
     */
    static EpisodeInfo run_env_episode(bboard::Environment& env, int maxSteps, bool printSteps=false, bool printFirstLast=false);

    /**
     * @brief run Run the environment with the given agents and optionally collect logs.
     * @param agents The agents which will be used in the environment.
     * @param gameMode The gamemode of the environment.
     * @param config The configuration for this run.
     */
    static void run(std::array<bboard::Agent*, bboard::AGENT_COUNT> agents, RunnerConfig config);

    /**
     * @brief run_simple_unbiased_agents Run the environment with simple unbiased agents and optionally collect logs.
     * @param config The configuration for this run.
     */
    static void run_simple_unbiased_agents(RunnerConfig config);

private:

};

#endif // RUNNER_H
