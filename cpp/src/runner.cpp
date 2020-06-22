#include "runner.h"

#include <iostream>
#include "log_agent.h"

#include "agents.hpp"

Runner::Runner()
{
    // TODO maybe create persistent environment and reset it?
}

EpisodeInfo Runner::run(bboard::Environment& env, int maxSteps)
{
    int step = 0;
    while (!env.IsDone() && step < maxSteps) {
        env.Step(false);
        step++;
    }

    EpisodeInfo info;
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

void _polulate_with_simple_agents(LogAgent* logAgents, int count) {
    for (int i = 0; i < count; i++) {
        logAgents[i].deleteAgent();
        logAgents[i].reset(new agents::SimpleAgent());
    }
}

void Runner::generateSupervisedTrainingData(IPCManager* ipcManager, int maxEpisodeSteps, long maxEpisodes, long maxTotalSteps) {
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
        bboard::Environment env;
        env.MakeGame(agents, true);

        // populate the log agents with simple agents
        _polulate_with_simple_agents(logAgents, 4);

        EpisodeInfo result = run(env, maxEpisodeSteps);

        ipcManager->writeEpisodeInfo(result);

        // write the episode logs
        for (int i = 0; i < 4 && (totalEpisodeSteps == -1 || totalEpisodeSteps < maxTotalSteps); i++) {
            LogAgent a = logAgents[i];
            ipcManager->writeAgentExperience(&a, result);
            totalEpisodeSteps += a.step;
        }

        std::cout << "Total steps: " << totalEpisodeSteps << " > Episode " << e << ": steps " << result.steps << ", winner " << result.winner << ", is draw "
                  << result.isDraw << ", is done " << result.isDone << std::endl;
    }
}



