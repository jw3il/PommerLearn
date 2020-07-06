#include "runner.h"

#include <iostream>
#include "log_agent.h"

#include "agents.hpp"

#include <thread>
#include <chrono>

Runner::Runner()
{
    // TODO maybe create persistent environment and reset it?
}

EpisodeInfo Runner::run(bboard::Environment& env, int maxSteps, bool printSteps)
{
    EpisodeInfo info;
    info.initialState = env.GetState();

    int step = 0;

    if (printSteps) {
        std::cout << "Step " << step << std::endl;
        bboard::PrintState(&env.GetState(), false);
        std::cout << std::endl;
    }

    while (!env.IsDone() && step < maxSteps) {
        env.Step(false);
        step++;

        if (printSteps) {
            std::cout << "Step " << step << std::endl;
            bboard::PrintState(&env.GetState(), false);
            std::cout << std::endl;
        }
    }

    info.winner = env.GetWinner();
    info.isDraw = env.IsDraw();
    info.isDone = env.IsDone();
    info.steps = step;

    bboard::AgentInfo* agentInfos = env.GetState().agents;
    for (int i = 0; i < bboard::AGENT_COUNT; i++) {
        info.dead[i] = agentInfos[i].dead;
    }

    return info;
}

void _polulate_with_simple_agents(LogAgent* logAgents, int count, long seed) {
    for (int i = 0; i < count; i++) {
        logAgents[i].deleteAgent();
        logAgents[i].reset(new agents::SimpleAgent(seed + i));
    }
}

void Runner::generateSupervisedTrainingData(IPCManager* ipcManager, int maxEpisodeSteps, long maxEpisodes, long maxTotalSteps, bool printSteps) {
    // create log agents to log the episodes
    LogAgent logAgents[4] = {
        LogAgent(maxEpisodeSteps),
        LogAgent(maxEpisodeSteps),
        LogAgent(maxEpisodeSteps),
        LogAgent(maxEpisodeSteps)
    };
    std::array<bboard::Agent*, 4> agents = {&logAgents[0], &logAgents[1], &logAgents[2], &logAgents[3]};

    long totalEpisodeSteps = 0;
    for (int e = 0; (maxEpisodes == -1 || e < maxEpisodes) && (maxTotalSteps == -1 || totalEpisodeSteps < maxTotalSteps); e++) {
        long seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        bboard::Environment env;
        env.MakeGame(agents, seed, true);

        // populate the log agents with simple agents
        _polulate_with_simple_agents(logAgents, 4, seed);

        EpisodeInfo result = run(env, maxEpisodeSteps, printSteps);

        ipcManager->writeEpisodeInfo(result);

        // write the episode logs
        for (int i = 0; i < 4 && (maxTotalSteps == -1 || totalEpisodeSteps < maxTotalSteps); i++) {
            LogAgent a = logAgents[i];
            ipcManager->writeAgentExperience(&a, result);
            totalEpisodeSteps += a.step;
        }

        std::cout << "Total steps: " << totalEpisodeSteps << " > Episode " << e << ": steps " << result.steps << ", winner "
                  << result.winner << ", is draw " << result.isDraw << ", is done " << result.isDone << std::endl;
        std::cout << " > Seed: 0x" << std::hex << seed << std::dec << std::endl;
    }
}



