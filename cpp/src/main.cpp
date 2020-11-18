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

#include "boost/program_options.hpp"

#include "clonable.h"

namespace po = boost::program_options;

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

void free_for_all_tourney(long maxGames, long targetedSamples, long maxSamples, IPCManager* ipcManager, std::string modelDir)
{
    // TODO: Decouple agent creation
    srand(time(0));

    StateConstants::init(false);
#ifdef TENSORRT
    TensorrtAPI netSingle(0, 1, modelDir, "float32");
#elif defined (TORCH)
    TorchAPI netSingle("cpu", 0, 1, modelDir);
#endif
    SearchSettings searchSettings;
    searchSettings.virtualLoss = 1;
    searchSettings.batchSize = 8;
    searchSettings.threads = 2;
    searchSettings.useTranspositionTable = false;
    searchSettings.multiPV = 1;
    searchSettings.virtualLoss = 1;
    searchSettings.nodePolicyTemperature = 1;
    searchSettings.dirichletEpsilon = 0.25f;

    vector<unique_ptr<NeuralNetAPI>> netBatches = create_new_net_batches(modelDir, searchSettings);
    PlaySettings playSettings;
    SearchLimits searchLimits;
    searchLimits.simulations = 100;
    // searchLimits.movetime = 100;
    // searchLimits.moveOverhead = 20;
    EvalInfo evalInfo;

    RawNetAgent rawNetAgent(&netSingle, &playSettings, true);
    MCTSAgent mctsAgent(&netSingle, netBatches, &searchSettings, &playSettings);

    bboard::Environment env;
    bboard::GameMode gameMode = bboard::GameMode::FreeForAll;

    // this is the state object of agent 0
    PommermanState pommermanState(0, gameMode);

    // partial observability
    bboard::ObservationParameters obsParams;
    obsParams.agentPartialMapView = true;
    obsParams.agentInfoVisibility = bboard::AgentInfoVisibility::OnlySelf;
    obsParams.exposePowerUps = false;
    obsParams.agentViewSize = 4;
    // pommermanState.set_partial_observability(&obsParams);

    // other agents used for planning
    std::array<Clonable<bboard::Agent>*, bboard::AGENT_COUNT> planningAgents = {
        new CopyClonable<bboard::Agent, agents::SimpleAgent>(agents::SimpleAgent(rand())),
        new CopyClonable<bboard::Agent, agents::SimpleAgent>(agents::SimpleAgent(rand())),
        new CopyClonable<bboard::Agent, agents::SimpleAgent>(agents::SimpleAgent(rand())),
        new CopyClonable<bboard::Agent, agents::SimpleAgent>(agents::SimpleAgent(rand())),
    };
    pommermanState.set_planning_agents(planningAgents);

    std::array<bboard::Agent*, bboard::AGENT_COUNT> agents = {
        new CrazyAraAgent(&mctsAgent, &pommermanState, &searchLimits, &evalInfo),
        // new CrazyAraAgent(&rawNetAgent, &pommermanState, &searchLimits, &evalInfo);
        new agents::SimpleAgent(rand()),
        new agents::SimpleAgent(rand()),
        new agents::SimpleAgent(rand()),
    };

    Runner::run(agents, gameMode, 800, maxGames, targetedSamples, maxSamples, -1, false, ipcManager);
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
    po::options_description configDesc("Available options");

    configDesc.add_options()
            ("help", "Print help message")

            // general options
            ("mode", po::value<std::string>()->default_value("ffa_sl"), "Available modes: ffa_sl, ffa_mcts")
            // stop training if:
            //   num_games > max_games
            ("max_games", po::value<int>()->default_value(10), "The max. number of generated games (ignored if -1)")
            //   || num_samples >= max_samples (hard cut)
            ("max_samples", po::value<int>()->default_value(-1), "The max. number of generated samples (ignored if -1)")
            //   || num_samples >= targeted_samples (soft cut, episode will still be added as long as num_samples < max_samples)
            ("targeted_samples", po::value<int>()->default_value(-1), "The targeted number of generated samples, fully includes the last episode (ignored if -1). ")

            // log options
            ("log", "If set, generate enough samples to fill a whole dataset (chunk_size * chunk_count samples)")
            ("file_prefix", po::value<std::string>()->default_value("./data"), "Set the filename prefix for the new datasets")
            ("chunk_size", po::value<int>()->default_value(1000), "Max. number of samples in a single file inside the dataset")
            ("chunk_count", po::value<int>()->default_value(100), "Max. number of chunks in a dataset")

            // value options
            ("discount_factor", po::value<float>()->default_value(1), "The discount factor used to assign values to individual steps (ignored if >= 1)")
            ("add_agent_vals", po::value<bool>()->default_value(false), "Whether to add weighted agent values in the value calculation")

            // mcts options
            ("model_dir", po::value<std::string>()->default_value("./model"), "The directory which contains the agent's model(s) for multiple batch sizes")
    ;

    po::variables_map configVals;
    po::store(po::parse_command_line(argc, argv, configDesc), configVals);
    po::notify(configVals);

    if (configVals.count("help")) {
        std::cout << configDesc << "\n";
        return 1;
    }

    // check whether we want to log the games
    std::unique_ptr<FileBasedIPCManager> ipcManager;
    int maxSamples = configVals["max_samples"].as<int>();
    if (configVals.count("log")) {
        // read the value config
        ValueConfig valConf;
        valConf.discountFactor = configVals["discount_factor"].as<float>();
        valConf.addWeightedAgentValues = configVals["add_agent_vals"].as<bool>();

        ipcManager = std::make_unique<FileBasedIPCManager>(configVals["file_prefix"].as<std::string>(), configVals["chunk_size"].as<int>(), configVals["chunk_count"].as<int>(), valConf);

        // fill at most one dataset
        maxSamples = min(maxSamples, configVals["chunk_size"].as<int>() * configVals["chunk_count"].as<int>());
    }

    int maxGames = configVals["max_games"].as<int>();
    int samples = configVals["targeted_samples"].as<int>();
    if (configVals["mode"].as<std::string>() == "ffa_sl") {
        Runner::run_simple_agents(800, maxGames, samples, maxSamples, -1, false, ipcManager.get());
    }
    else if (configVals["mode"].as<std::string>() == "ffa_mcts") {
        free_for_all_tourney(maxGames, samples, maxSamples, ipcManager.get(), configVals["model_dir"].as<std::string>());
    }
    else {
        std::cerr << "Unknown mode" << std::endl;
    }

    if(ipcManager.get() != nullptr)
    {
        ipcManager->flush();
    }

    return 0;
}
