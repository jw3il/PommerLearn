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

EpisodeInfo Runner::run_env_episode(bboard::Environment& env, int maxSteps, bool printSteps)
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

void Runner::run(std::array<bboard::Agent*, bboard::AGENT_COUNT> agents, bboard::GameMode gameMode, int maxEpisodeSteps, long maxEpisodes, long targetedLoggedSteps, long maxLoggedSteps, long seed, bool printSteps, IPCManager* ipcManager) {
    if(seed == -1)
    {
        seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    }
    auto rng = std::mt19937_64(seed);

    // create the sample buffers for all agents wanting to collect samples
    std::vector<SampleCollector*> sampleCollectors;
    if (ipcManager != nullptr) {
        for (uint i = 0; i < agents.size(); i++) {
            SampleCollector* sampleCollector = dynamic_cast<SampleCollector*>(agents[i]);
            if (sampleCollector != nullptr) {
                sampleCollector->create_buffer(maxEpisodeSteps);
                sampleCollectors.push_back(sampleCollector);
            }
        }
    }

    std::cout << "Number of logging agents: " << sampleCollectors.size() << std::endl;

    int nbNotDone = 0;
    int nbDraws = 0;
    std::array<int, bboard::AGENT_COUNT> nbWins;
    std::fill(nbWins.begin(), nbWins.end(), 0);

    long totalLoggedSteps = 0;
    int episode = 0;
    for (; (maxEpisodes == -1 || episode < maxEpisodes)
         && (maxLoggedSteps == -1 || totalLoggedSteps < maxLoggedSteps)
         && (targetedLoggedSteps == -1 || totalLoggedSteps < targetedLoggedSteps); episode++) {

        // generate new seeds in every episode
        seed = rng();
        bboard::Environment env;
        env.MakeGame(agents, gameMode, seed, true);

        EpisodeInfo result = Runner::run_env_episode(env, maxEpisodeSteps, printSteps);

        if (ipcManager != nullptr) {
            ipcManager->writeNewEpisode(result);

            // write the episode logs (if there are any logging agents)
            for (SampleCollector* collector : sampleCollectors) {
                if ((maxLoggedSteps > 0 && totalLoggedSteps >= maxLoggedSteps))
                    break;

                int steps;
                if (maxLoggedSteps == -1) {
                    steps = ipcManager->writeAgentExperience(collector);
                }
                else {
                    steps = ipcManager->writeAgentExperience(collector, maxLoggedSteps - totalLoggedSteps);
                }
                totalLoggedSteps += steps;

                collector->get_buffer()->clear();
            }
        }

        if (!result.isDone) {
            nbNotDone++;
        }
        else if (result.isDraw) {
            nbDraws++;
        }
        else {
            // add winning agents, this also works for teams
            for (int i = 0; i < bboard::AGENT_COUNT; i++) {
                if (env.GetState().agents[i].won) {
                    nbWins[i] += 1;
                }
            }
        }

        std::cout << "Total logged steps: " << totalLoggedSteps << " > Episode " << episode << ": steps " << result.steps << ", ";
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

    // display aggregated statistics

    std::cout << "------------------------------" << std::endl;

    std::cout << "Total episodes: " << episode <<  std::endl;
    std::cout << "Wins: " << std::endl;
    for (size_t agentIdx = 0; agentIdx < bboard::AGENT_COUNT; ++agentIdx) {
        int wins = nbWins[agentIdx];
        std::cout << "- Agent " << agentIdx << ": " << wins << " (" << (float)wins * 100 / episode << "%)" << std::endl;
    }

    std::cout << "Draws: " << nbDraws << " (" << (float)nbDraws * 100 / episode << "%)" << std::endl;
    std::cout << "Not done: " << nbNotDone << " (" << (float)nbNotDone * 100 / episode << "%)" << std::endl;
}

void Runner::run_simple_agents(int maxEpisodeSteps, long maxEpisodes, long targetedLoggedSteps, long maxLoggedSteps, long seed, bool printSteps, IPCManager* ipcManager) {
    // create wrappers to log the actions of some agents
    WrappedLogAgent agentWrappers[4];
    std::array<bboard::Agent*, 4> agents = {&agentWrappers[0], &agentWrappers[1], &agentWrappers[2], &agentWrappers[3]};

    // make the wrappers act like simple agents
    for (int i = 0; i < 4; i++) {
        agentWrappers[i].set_agent(std::make_unique<agents::SimpleAgent>(seed + i));
    }

    Runner::run(agents, bboard::GameMode::FreeForAll, maxEpisodeSteps, maxEpisodes, targetedLoggedSteps, maxLoggedSteps, seed, printSteps, ipcManager);

    for (int i = 0; i < 4; i++) {
        agentWrappers[i].release_agent();
    }
}



