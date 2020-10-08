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

void log_episodes(IPCManager* ipcManager, std::array<bboard::Agent*, bboard::AGENT_COUNT> agents, bboard::GameMode gameMode, int maxEpisodeSteps, long maxEpisodes, long maxTotalSteps, long seed, bool printSteps) {
    if(seed == -1)
    {
        seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    }
    auto rng = std::mt19937_64(seed);

    // create the sample buffers for all agents wanting to collect samples
    std::vector<SampleCollector*> sampleCollectors;
    for (uint i = 0; i < agents.size(); i++) {
        SampleCollector* sampleCollector = dynamic_cast<SampleCollector*>(agents[i]);
        if (sampleCollector != nullptr) {
            sampleCollector->create_buffer(maxEpisodeSteps);
            sampleCollectors.push_back(sampleCollector);
        }
    }

    std::cout << "Number of logging agents: " << sampleCollectors.size() << std::endl;

    if (sampleCollectors.size() == 0) {
        std::cerr << "No logging agents detected! No actions will be logged!" << std::endl;
    }

    long totalEpisodeSteps = 0;
    for (int e = 0; (maxEpisodes == -1 || e < maxEpisodes) && (maxTotalSteps == -1 || totalEpisodeSteps < maxTotalSteps); e++) {
        // generate new seeds in every episode
        seed = rng();
        bboard::Environment env;
        env.MakeGame(agents, gameMode, seed, true);

        EpisodeInfo result = Runner::run(env, maxEpisodeSteps, printSteps);

        ipcManager->writeNewEpisode(result);

        // write the episode logs (if there are any logging agents)
        for (SampleCollector* collector : sampleCollectors) {
            if ((maxTotalSteps > 0 && totalEpisodeSteps >= maxTotalSteps))
                break;

            int steps = ipcManager->writeAgentExperience(collector);
            totalEpisodeSteps += steps;

            collector->get_buffer()->clear();
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

    // TODO: display aggregated statistics
}

void Runner::generateSupervisedTrainingData(IPCManager* ipcManager, int maxEpisodeSteps, long maxEpisodes, long maxTotalSteps, long seed, bool printSteps) {
    // create wrappers to log the actions of some agents
    WrappedLogAgent agentWrappers[4];
    std::array<bboard::Agent*, 4> agents = {&agentWrappers[0], &agentWrappers[1], &agentWrappers[2], &agentWrappers[3]};

    // make the wrappers act like simple agents
    for (int i = 0; i < 4; i++) {
        agentWrappers[i].set_agent(std::make_unique<agents::SimpleAgent>(seed + i));
    }

    log_episodes(ipcManager, agents, bboard::GameMode::FreeForAll, maxEpisodeSteps, maxEpisodes, maxTotalSteps, seed, printSteps);

    for (int i = 0; i < 4; i++) {
        agentWrappers[i].release_agent();
    }
}



