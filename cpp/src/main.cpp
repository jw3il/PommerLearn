#include <iostream>
#include <array>

#include "runner.h"
#include "ipc_manager.h"
#include "nn/neuralnetapi.h"
#include "nn/tensorrtapi.h"
#include "agents/rawnetagent.h"
#include "pommerman_raw_net_agent.h"
#include "stateobj.h"

#include "agents.hpp"

void load_models()
{
    make_unique<TensorrtAPI>(1, 1, "model/", "float32");
    make_unique<TensorrtAPI>(1, 1, "model/", "float16");
    make_unique<TensorrtAPI>(1, 1, "model/", "int8");

    make_unique<TensorrtAPI>(1, 8, "model/", "float32");
    make_unique<TensorrtAPI>(1, 8, "model/", "float16");
    make_unique<TensorrtAPI>(1, 8, "model/", "int8");
}

void free_for_all_tourney(size_t nbGames)
{
    Constants::init(false);
    TensorrtAPI nn(0, 1, "model", "float32");
    PlaySettings playSettings;
    RawNetAgent rawNetAgent(&nn, &playSettings, true);

    bboard::Environment env;
    bboard::GameMode gameMode = bboard::GameMode::FreeForAll;

    bboard::ObservationParameters obsParams;
    obsParams.agentPartialMapView = true;
    obsParams.agentInfoVisibility = bboard::AgentInfoVisibility::OnlySelf;
    obsParams.exposePowerUps = false;
    obsParams.agentViewSize = 4;

    // this is the state object of agent 0
    PommermanState pommermanState(0, gameMode);
    // pommermanState.set_partial_observability(&obsParams);
    PommermanRawNetAgent pommerRawAgent(&rawNetAgent, &pommermanState);

    srand(time(0));
    std::array<bboard::Agent*, bboard::AGENT_COUNT> agents = {&pommerRawAgent,
                                                              new agents::SimpleAgent(rand()),
                                                              new agents::SimpleAgent(rand()),
                                                              new agents::SimpleAgent(rand()),
                                                             };
    std::array<size_t, 4> nbWins = {0,0,0,0};
    size_t nbDraws = 0;

    for (size_t curIt = 0; curIt < nbGames; ++curIt) {
        env.MakeGame(agents, gameMode, rand(), true);
        env.RunGame(800);

        const bboard::State& lastState = env.GetState();
        if (lastState.winningAgent != -1) {
            nbWins[lastState.winningAgent] += 1;
        }
        else {
            ++nbDraws;
        }
    }

    for (size_t agentIdx = 0; agentIdx < bboard::AGENT_COUNT; ++agentIdx) {
        std::cout << "Agent " << agentIdx << ": " << nbWins[agentIdx] << " wins" << std::endl;
    }

    std::cout << "Draws: " << nbDraws << std::endl;
}

void generate_sl_data(const std::string& dataPrefix, int chunkSize, int chunkCount)
{
    FileBasedIPCManager ipcManager(dataPrefix, chunkSize, chunkCount);
    Runner runner;

    // generate enough steps (chunkSize * chunkCount) to fill one dataset
    runner.generateSupervisedTrainingData(&ipcManager, 800, -1, chunkSize * chunkCount, -1, false);
    ipcManager.flush();
}

int main(int argc, char **argv) {
    std::string dataPrefix = "data";
    if (argc >= 2) {
        dataPrefix = argv[1];
    }

    // generate_sl_data(dataPrefix, 1000, 100);
    free_for_all_tourney(100);

    return 0;
}
