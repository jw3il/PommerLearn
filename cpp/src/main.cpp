#include <iostream>
#include <array>

#include "runner.h"
#include "ipc_manager.h"
#include "nn/neuralnetapi.h"
#include "nn/tensorrtapi.h"
#include "nn/torchapi.h"
#include "agents/rawnetagent.h"
#include "agents/mctsagent.h"
#include "crazyara_agent.h"
#include "stateobj.h"

#include "agents.hpp"

void load_models()
{
#ifdef TENSORRT
    make_unique<TensorrtAPI>(1, 1, "model/", "float32");
    make_unique<TensorrtAPI>(1, 1, "model/", "float16");
    make_unique<TensorrtAPI>(1, 1, "model/", "int8");

    make_unique<TensorrtAPI>(1, 8, "model/", "float32");
    make_unique<TensorrtAPI>(1, 8, "model/", "float16");
    make_unique<TensorrtAPI>(1, 8, "model/", "int8");
#elif defined (TORCH)
    make_unique<TorchAPI>("cpu", 0, 1, "model/");
#endif
}

vector<unique_ptr<NeuralNetAPI>> create_new_net_batches(const string& modelDirectory, const SearchSettings& searchSettings)
{
    vector<unique_ptr<NeuralNetAPI>> netBatches;
#ifdef MXNET
    #ifdef TENSORRT
        const bool useTensorRT = bool(Options["Use_TensorRT"]);
    #else
        const bool useTensorRT = false;
    #endif
#endif
    int First_Device_ID = 0;
    int Last_Device_ID = 0;
    for (int deviceId = First_Device_ID; deviceId <= Last_Device_ID; ++deviceId) {
        for (size_t i = 0; i < searchSettings.threads; ++i) {
    #ifdef MXNET
            netBatches.push_back(make_unique<MXNetAPI>(Options["Context"], deviceId, searchSettings.batchSize, modelDirectory, useTensorRT));
    #elif defined TENSORRT
            netBatches.push_back(make_unique<TensorrtAPI>(deviceId, searchSettings.batchSize, modelDirectory, "float16"));
    #elif defined TORCH
            netBatches.push_back(make_unique<TorchAPI>("cpu", deviceId, searchSettings.batchSize, modelDirectory));
    #endif
        }
    }
    return netBatches;
}

void free_for_all_tourney(size_t nbGames)
{
    StateConstants::init(false);
#ifdef TENSORRT
    TensorrtAPI netSingle(0, 1, "model", "float32");
#elif defined (TORCH)
    TorchAPI netSingle("cpu", 0, 1, "model/");
#endif
    SearchSettings searchSettings;
    searchSettings.virtualLoss = 1;
    searchSettings.batchSize = 8;
    searchSettings.threads = 1;
    searchSettings.useTranspositionTable = false;
    searchSettings.multiPV = 1;
    searchSettings.nodePolicyTemperature = 1;
    searchSettings.dirichletEpsilon = 0.2;

    vector<unique_ptr<NeuralNetAPI>> netBatches = create_new_net_batches("model", searchSettings);
    PlaySettings playSettings;
    SearchLimits searchLimits;
    searchLimits.movetime = 100;
    searchLimits.moveOverhead = 20;
    EvalInfo evalInfo;

    RawNetAgent rawNetAgent(&netSingle, &playSettings, true);
    MCTSAgent mctsAgent(&netSingle, netBatches, &searchSettings, &playSettings);

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

    CrazyAraAgent crazyaraAgent(&mctsAgent, &pommermanState, &searchLimits, &evalInfo);
//    CrazyAraAgent crazyaraAgent(&rawNetAgent, &pommermanState, &searchLimits, &evalInfo);

    srand(time(0));
    std::array<bboard::Agent*, bboard::AGENT_COUNT> agents = {&crazyaraAgent,
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
    free_for_all_tourney(10);

    return 0;
}
