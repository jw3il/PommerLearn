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

void free_for_all_tourney(size_t nbGames, IPCManager* ipcManager)
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

    srand(time(0));

    std::array<bboard::Agent*, bboard::AGENT_COUNT> agents = {
        new CrazyAraAgent(&mctsAgent, &pommermanState, &searchLimits, &evalInfo),
        // new CrazyAraAgent(&rawNetAgent, &pommermanState, &searchLimits, &evalInfo);
        new agents::SimpleAgent(rand()),
        new agents::SimpleAgent(rand()),
        new agents::SimpleAgent(rand()),
    };

    Runner::run(agents, gameMode, 800, nbGames, -1, -1, false, ipcManager);
}

void generate_sl_data(int nbSamples, IPCManager* ipcManager)
{
    if (ipcManager == nullptr) {
        std::cout << "Cannot generate SL data without an IPCManager instance!" << std::endl;
        return;
    }

    Runner::run_simple_agents(800, -1, nbSamples, -1, false, ipcManager);
}

int main(int argc, char **argv) {
    std::string dataPrefix = "data";
    if (argc >= 2) {
        dataPrefix = argv[1];
    }

    int chunkSize = 1000;
    int chunkCount = 100;

    std::unique_ptr<FileBasedIPCManager> ipcManager;

    // uncomment to enable logging
    // ipcManager = std::make_unique<FileBasedIPCManager>(dataPrefix, chunkSize, chunkCount);

    // generate enough steps to fill one dataset
    // generate_sl_data(chunkCount * chunkSize, ipcManager.get());

    free_for_all_tourney(10, ipcManager.get());

    if(ipcManager.get() != nullptr)
    {
        ipcManager->flush();
    }

    return 0;
}
