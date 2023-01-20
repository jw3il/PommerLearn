#include "runner.h"

#include <iostream>
#include "agents/log_agent.h"
#include "agents/virtual_step_wrapper.h"
#include "agents/crazyara_agent.h"

#include "agents.hpp"

#include <thread>
#include <chrono>

Runner::Runner()
{
    // TODO maybe create persistent environment and reset it?
}

EpisodeInfo Runner::run_env_episode(bboard::Environment& env, int maxSteps, bool printSteps, bool printFirstLast)
{
    EpisodeInfo info;
    info.initialState = env.GetState();
    info.gameMode = env.GetGameMode();

    if (printSteps || printFirstLast) {
        env.Print(false);
    }

    const bboard::State& currentState = env.GetState();
    int startSteps = currentState.timeStep;
    while (!env.IsDone() && (maxSteps <= 0 || currentState.timeStep - startSteps < maxSteps))
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

    if (printFirstLast && !printSteps) {
        env.Print(false);
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

void _print_stats(std::array<bboard::Agent*, bboard::AGENT_COUNT> agents, std::chrono::steady_clock::time_point begin, int episode, long totalEpisodeSteps, int nbNotDone, int nbDraws, std::array<int, bboard::AGENT_COUNT> nbWins, std::array<int, bboard::AGENT_COUNT> nbAlive, bool csv)
{
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    double elapsedSeconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / 1000.0;

    // print as human-readable format
    std::cout << "------------------------------" << std::endl;
    std::cout << "Total episodes: " << episode <<  std::endl;
    std::cout << "Average steps: " << totalEpisodeSteps / episode <<  std::endl;
    std::cout << "Wins | Alive: " << std::endl;
    for (size_t agentIdx = 0; agentIdx < bboard::AGENT_COUNT; ++agentIdx) {
        int wins = nbWins[agentIdx];
        int alive = nbAlive[agentIdx];
        std::cout << "- Agent " << agentIdx << ": " << wins << " (" << (float)wins * 100 / episode << "%) | " << alive << " (" << (float)alive * 100 / episode << "%)"; 
        CrazyAraAgent* crazyAraAgent = dynamic_cast<CrazyAraAgent*>(agents[agentIdx]);
        if (crazyAraAgent) {
            std::ostringstream stream;
            stream << std::fixed << std::setprecision(2);
            stream << " t: " << crazyAraAgent->eval_time_ms.get_mean() << " ms, n: " << crazyAraAgent->eval_nodes.get_mean() << ", d: " << crazyAraAgent->eval_depth.get_mean();
            std::cout << stream.str();
        }
        std::cout << std::endl;
    }

    std::cout << "Draws: " << nbDraws << " (" << (float)nbDraws * 100 / episode << "%)" << std::endl;
    std::cout << "Not done: " << nbNotDone << " (" << (float)nbNotDone * 100 / episode << "%)" << std::endl;
    std::cout << "Elapsed time: " << elapsedSeconds << " s (" << elapsedSeconds / episode << " s/episode, " << (elapsedSeconds * 1000) / totalEpisodeSteps << " ms/step)" << std::endl;
    std::cout << "------------------------------" << std::endl;

    if (csv) {
        // print stats in machine-readable format
        std::cout << episode << "," << totalEpisodeSteps;
        std::string evalString = "";
        for (size_t agentIdx = 0; agentIdx < bboard::AGENT_COUNT; ++agentIdx) {
            int wins = nbWins[agentIdx];
            int alive = nbAlive[agentIdx];
            std::cout << "," << wins << "," << alive;
            CrazyAraAgent* crazyAraAgent = dynamic_cast<CrazyAraAgent*>(agents[agentIdx]);
            if (crazyAraAgent) {
                if (evalString != "") {
                    // agent separator
                    evalString += "|";
                }
                // print all stats
                evalString += crazyAraAgent->eval_time_ms.get_csv() + ":" + crazyAraAgent->eval_nodes.get_csv() + ":" + crazyAraAgent->eval_depth.get_csv();
            }
        }
        std::cout << "," << nbDraws << "," << nbNotDone << "," << elapsedSeconds << ",\"" << evalString << "\"" << std::endl;
    }
}

void Runner::run(std::array<bboard::Agent*, bboard::AGENT_COUNT> agents, RunnerConfig config) {
    // create the sample buffers for all agents wanting to collect samples
    std::vector<SampleCollector*> sampleCollectors;
    if (config.ipcManager != nullptr) {
        for (uint i = 0; i < agents.size(); i++) {
            SampleCollector* sampleCollector = dynamic_cast<SampleCollector*>(agents[i]);
            if (sampleCollector != nullptr && sampleCollector->get_logging_enabled()) {
                sampleCollector->create_buffer(config.maxEpisodeSteps);
                sampleCollectors.push_back(sampleCollector);
            }
        }
    }

    std::cout << "Number of logging agents: " << sampleCollectors.size() << std::endl;

    bboard::Environment env;
    if (config.useStateInSearch) {
        for (int i = 0; i < bboard::AGENT_COUNT; i++) {
            CrazyAraAgent* crazyAraAgent = dynamic_cast<CrazyAraAgent*>(agents[i]);
            if (crazyAraAgent) {
                std::cout << "Info: Agent " << i << " has access to the state." << std::endl;
                crazyAraAgent->use_environment_state(&env);
            }
        }
    }

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    int nbNotDone = 0;
    int nbDraws = 0;
    std::array<int, bboard::AGENT_COUNT> nbWins;
    std::fill(nbWins.begin(), nbWins.end(), 0);
    std::array<int, bboard::AGENT_COUNT> nbAlive;
    std::fill(nbAlive.begin(), nbAlive.end(), 0);

    auto boardRNG = std::mt19937_64(config.seed);
    auto agentPositionRNG = std::mt19937_64(config.seed);

    long currentEnvSeed = config.envSeed;
    long currentAgentPositionSeed = -1;

    long totalEpisodeSteps = 0;
    long totalLoggedSteps = 0;
    int episode = 0;
    int nextEnvSeedEps = 0;
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

        if (config.randomAgentPositions) {
            currentAgentPositionSeed = agentPositionRNG();
        }

        env.MakeGame(agents, config.gameMode, currentEnvSeed, currentAgentPositionSeed);
        env.SetObservationParameters(config.observationParameters);

        EpisodeInfo result = Runner::run_env_episode(env, config.maxEpisodeSteps, config.printSteps, config.printFirstLast);

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

            if(config.flushEpisodes >= 0 && (episode + 1) % config.flushEpisodes == 0)
            {
                config.ipcManager->flush();
            }
        }

        if (!result.isDone) {
            nbNotDone++;
        }
        else if (result.isDraw) {
            nbDraws++;
        }

        // count winning & alive agents, this also works for teams
        for (int i = 0; i < bboard::AGENT_COUNT; i++) {
            const bboard::State& state = env.GetState();
            if (state.IsWinner(i)) {
                nbWins[i] += 1;
            }
            if (!state.agents[i].dead) {
                nbAlive[i] += 1;
            }
        }

        std::cout << "Total logged steps: " << totalLoggedSteps << " > Episode " << episode << ": steps " << result.steps << ", ";
        if (result.winningAgent != -1)
        {
            std::cout << "winning agent " << result.winningAgent;
        }
        else if (result.winningTeam > 0)
        {
            std::cout << "winning team " << result.winningTeam;
        }
        else
        {
            std::cout << "no winner";
        }
        std::cout << ", is draw " << result.isDraw << ", is done " << result.isDone << std::endl;

        std::cout << " > Seed: 0x" << std::hex << currentEnvSeed << std::dec << std::endl;

        if ((episode + 1) % 20 == 0)
        {
            _print_stats(agents, begin, episode + 1, totalEpisodeSteps, nbNotDone, nbDraws, nbWins, nbAlive, false);
        }
    }

    // display aggregated statistics
    _print_stats(agents, begin, episode, totalEpisodeSteps, nbNotDone, nbDraws, nbWins, nbAlive, true);
}

void Runner::run_simple_unbiased_agents(RunnerConfig config) {
    // create wrappers to log the actions of some agents
    WrappedLogAgent* agentWrappers[4];    
    for (int i = 0; i < 4; i++) {
        if (config.useVirtualStep) {
            agentWrappers[i] = new VirtualStepWrapper<WrappedLogAgent>();
        }
        else {
            agentWrappers[i] = new WrappedLogAgent();
        }

        // make the wrappers act like regular agents
        agentWrappers[i]->set_agent(std::make_unique<agents::SimpleUnbiasedAgent>(config.seed + i));
    }

    std::array<bboard::Agent*, 4> agents = {agentWrappers[0], agentWrappers[1], agentWrappers[2], agentWrappers[3]};
    Runner::run(agents, config);

    for (int i = 0; i < 4; i++) {
        agentWrappers[i]->release_agent();
        delete agentWrappers[i];
    }
}
