#include "runner.h"

#include <iostream>
#include "log_agent.h"

#include "agents.hpp"

Runner::Runner()
{
    // TODO maybe create persistent environment and reset it?
}

EpisodeInfo Runner::run(std::array<bboard::Agent*, 4> agents, int maxSteps)
{
    bboard::Environment env;
    env.MakeGame(agents, true);

    int step = 0;
    while (!env.IsDone() && step < maxSteps) {
        env.Step(false);
        step++;
    }

    return (EpisodeInfo){env.GetWinner(), step};
}

void _polulate_with_simple_agents(LogAgent* logAgents, int count) {
    for (int i = 0; i < count; i++) {
        logAgents[i].deleteAgent();
        logAgents[i].reset(new agents::SimpleAgent());
    }
}

void Runner::generateSupervisedTrainingData(IPCManager* ipcManager, int maxSteps, int episodes) {
    // create log agents to log the episodes
    LogAgent logAgents[4] = {
        LogAgent(maxSteps),
        LogAgent(maxSteps),
        LogAgent(maxSteps),
        LogAgent(maxSteps)
    };
    std::array<bboard::Agent*, 4> agents = {&logAgents[0], &logAgents[1], &logAgents[2], &logAgents[3]};

    for (int e = 0; e < episodes; e++) {
        // pupulate the log agents with simple agents
        _polulate_with_simple_agents(logAgents, 4);

        EpisodeInfo result = run(agents, maxSteps);

        std::cout << "Episode: " << e << ": " << result.steps << ", " << result.winner << std::endl;

        // write the episode logs
        for (int i = 0; i < 4; i++) {
            LogAgent a = logAgents[i];
            a.won = (result.winner == a.id);

            ipcManager->writeEpisode(&a);
        }
    }
}



