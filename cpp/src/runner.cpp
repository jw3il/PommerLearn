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

    const bboard::State& currentState = env.GetState();
    int startSteps = currentState.timeStep;
    while(!env.IsDone() && (maxSteps <= 0 || currentState.timeStep - startSteps < maxSteps))
    {
        // execute the step
        env.Step(false);
        if(printSteps)
        {
            std::cout << "Step: " << currentState.timeStep << std::endl;
            env.Print(false);
        }

        // log actions
        for(int i = 0; i < bboard::AGENT_COUNT; i++)
        {
            if(env.HasActed(i))
            {
                info.actions[i].push_back((int8_t)env.GetLastMove(i));
            }
        }
    }

    info.winningTeam = env.GetWinningTeam();
    info.winningAgent = env.GetWinningAgent();

    info.isDraw = env.IsDraw();
    info.isDone = env.IsDone();
    info.steps = env.GetState().timeStep;

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

void Runner::generateSupervisedTrainingData(IPCManager* ipcManager, int maxEpisodeSteps, long maxEpisodes, long maxTotalSteps, long seed, bool printSteps) {
    // create log agents to log the episodes
    LogAgent logAgents[4] = {
        LogAgent(maxEpisodeSteps),
        LogAgent(maxEpisodeSteps),
        LogAgent(maxEpisodeSteps),
        LogAgent(maxEpisodeSteps)
    };
    std::array<bboard::Agent*, 4> agents = {&logAgents[0], &logAgents[1], &logAgents[2], &logAgents[3]};

    if(seed == -1)
    {
        seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    }
    auto rng = std::mt19937_64(seed);

    long totalEpisodeSteps = 0;
    for (int e = 0; (maxEpisodes == -1 || e < maxEpisodes) && (maxTotalSteps == -1 || totalEpisodeSteps < maxTotalSteps); e++) {
        // generate new seeds in every episode
        seed = rng();
        bboard::Environment env;
        env.MakeGame(agents, bboard::GameMode::FreeForAll, seed, true);

        // populate the log agents with simple agents
        _polulate_with_simple_agents(logAgents, 4, seed);

        EpisodeInfo result = run(env, maxEpisodeSteps, printSteps);

        ipcManager->writeNewEpisode(result);

        // write the episode logs
        for (int i = 0; i < 4 && (maxTotalSteps == -1 || totalEpisodeSteps < maxTotalSteps); i++) {
            LogAgent& a = logAgents[i];
            ipcManager->writeAgentExperience(&a);
            totalEpisodeSteps += a.step;
        }

        std::cout << "Total steps: " << totalEpisodeSteps << " > Episode " << e << ": steps " << result.steps << ", ";
        if (result.winningAgent != -1)
        {
            std::cout << "winning agent " << result.winningAgent;
        }
        else if (result.winningTeam != 0)
        {
            std::cout << "winning team " << result.winningTeam;
        }
        else
        {
            std::cout << "no winner";
        }
        std::cout << ", is draw " << result.isDraw << ", is done " << result.isDone << std::endl;

        std::cout << " > Seed: 0x" << std::hex << seed << std::dec << std::endl;
    }
}



