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

void Runner::run(std::array<bboard::Agent*, bboard::AGENT_COUNT> agents, bboard::GameMode gameMode, RunnerConfig config) {
    // create the sample buffers for all agents wanting to collect samples
    std::vector<SampleCollector*> sampleCollectors;
    if (config.ipcManager != nullptr) {
        for (uint i = 0; i < agents.size(); i++) {
            SampleCollector* sampleCollector = dynamic_cast<SampleCollector*>(agents[i]);
            if (sampleCollector != nullptr) {
                sampleCollector->create_buffer(config.maxEpisodeSteps);
                sampleCollectors.push_back(sampleCollector);
            }
        }
    }

    std::cout << "Number of logging agents: " << sampleCollectors.size() << std::endl;

    int nbNotDone = 0;
    int nbDraws = 0;
    std::array<int, bboard::AGENT_COUNT> nbWins;
    std::fill(nbWins.begin(), nbWins.end(), 0);

    auto boardRNG = std::mt19937_64(config.seed);

    long totalEpisodeSteps = 0;
    long totalLoggedSteps = 0;
    int episode = 0;
    int nextEnvSeedEps = 0;
    int currentEnvSeed = config.envSeed;
    for (; (config.maxEpisodes == -1 || episode < config.maxEpisodes)
         && (config.maxLoggedSteps == -1 || totalLoggedSteps < config.maxLoggedSteps)
         && (config.targetedLoggedSteps == -1 || totalLoggedSteps < config.targetedLoggedSteps); episode++) {

        // only generate random environments if envSeed == -1
        if (config.envSeed == -1) {
            // generate a new environment seed every envGenSeedEps episodes
            nextEnvSeedEps--;
            if (nextEnvSeedEps <= 0) {
                currentEnvSeed = boardRNG();
                nextEnvSeedEps = config.envGenSeedEps;
            }
        }

        bboard::Environment env;
        env.MakeGame(agents, gameMode, currentEnvSeed, currentEnvSeed);

        EpisodeInfo result = Runner::run_env_episode(env, config.maxEpisodeSteps, config.printSteps);

        totalEpisodeSteps += result.steps;

        if (config.ipcManager != nullptr) {
            config.ipcManager->writeNewEpisode(result);

            // write the episode logs (if there are any logging agents)
            for (SampleCollector* collector : sampleCollectors) {
                if ((config.maxLoggedSteps > 0 && totalLoggedSteps >= config.maxLoggedSteps))
                    break;

                int steps;
                if (config.maxLoggedSteps == -1) {
                    steps = config.ipcManager->writeAgentExperience(collector);
                }
                else {
                    steps = config.ipcManager->writeAgentExperience(collector, config.maxLoggedSteps - totalLoggedSteps);
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

        std::cout << " > Seed: 0x" << std::hex << currentEnvSeed << std::dec << std::endl;
    }

    // display aggregated statistics

    std::cout << "------------------------------" << std::endl;

    std::cout << "Total episodes: " << episode <<  std::endl;
    std::cout << "Average steps: " << totalEpisodeSteps / episode <<  std::endl;
    std::cout << "Wins: " << std::endl;
    for (size_t agentIdx = 0; agentIdx < bboard::AGENT_COUNT; ++agentIdx) {
        int wins = nbWins[agentIdx];
        std::cout << "- Agent " << agentIdx << ": " << wins << " (" << (float)wins * 100 / episode << "%)" << std::endl;
    }

    std::cout << "Draws: " << nbDraws << " (" << (float)nbDraws * 100 / episode << "%)" << std::endl;
    std::cout << "Not done: " << nbNotDone << " (" << (float)nbNotDone * 100 / episode << "%)" << std::endl;
}

void Runner::run_simple_agents(RunnerConfig config) {
    // create wrappers to log the actions of some agents
    WrappedLogAgent agentWrappers[4];
    std::array<bboard::Agent*, 4> agents = {&agentWrappers[0], &agentWrappers[1], &agentWrappers[2], &agentWrappers[3]};

    // make the wrappers act like simple agents
    for (int i = 0; i < 4; i++) {
        agentWrappers[i].set_agent(std::make_unique<agents::SimpleAgent>(config.seed + i));
    }

    Runner::run(agents, bboard::GameMode::FreeForAll, config);

    for (int i = 0; i < 4; i++) {
        agentWrappers[i].release_agent();
    }
}



