#include <iostream>
#include <array>
#include <chrono>

#include "runner.h"
#include "ipc_manager.h"
#include "nn/neuralnetapi.h"
#include "nn/tensorrtapi.h"
#include "nn/torchapi.h"
#include "crazyara_agent.h"
#include "stateobj.h"

#include "agents.hpp"

#include "boost/program_options.hpp"

#include "clonable.h"

namespace po = boost::program_options;

void free_for_all_tourney(std::string modelDir, RunnerConfig config, bool useRawNet, uint stateSize, uint valueVersion, PlanningAgentType planningAgentType, SearchLimits searchLimits)
{
    StateConstants::init(false);
    StateConstantsPommerman::set_auxiliary_outputs(stateSize);

    bboard::GameMode gameMode = bboard::GameMode::FreeForAll;

    std::cout << "Loading agents.." << std::endl;
    std::unique_ptr<CrazyAraAgent> crazyAraAgent;
    if (useRawNet)
    {
        crazyAraAgent = std::make_unique<CrazyAraAgent>(modelDir);
    }
    else {
        SearchSettings searchSettings = CrazyAraAgent::get_default_search_settings(true);
        PlaySettings playSettings;
        crazyAraAgent = std::make_unique<CrazyAraAgent>(modelDir, playSettings, searchSettings, searchLimits);
    }

    // partial observability
    bboard::ObservationParameters obsParams;
    obsParams.agentPartialMapView = false;
    obsParams.agentInfoVisibility = bboard::AgentInfoVisibility::All;
    obsParams.exposePowerUps = false;
    obsParams.agentViewSize = 4;

    crazyAraAgent->init_state(gameMode, obsParams, valueVersion, planningAgentType);

    srand(config.seed);
    std::array<bboard::Agent*, bboard::AGENT_COUNT> agents = {
        crazyAraAgent.get(),
        new agents::SimpleUnbiasedAgent(rand()),
        new agents::SimpleUnbiasedAgent(rand()),
        new agents::SimpleUnbiasedAgent(rand()),
    };

    std::cout << "Agents loaded. Starting the runner.." << std::endl;
    Runner::run(agents, gameMode, config);
}

int main(int argc, char **argv) {
    po::options_description configDesc("Available options");

    configDesc.add_options()
            ("help", "Print help message")

            // general options
            ("mode", po::value<std::string>()->default_value("ffa_sl"), "Available modes: ffa_sl, ffa_mcts")
            ("print", "If set, print the current state of the environment in every step.")
            ("print_first_last", "If set, print the first and last environment state of each episode.")

            // seeds and environment generation
            ("env_seed", po::value<long>()->default_value(-1), "The seed used for environment generation (= fixed environment in all episodes, ignored if -1)")
            ("env_gen_seed_eps", po::value<long>()->default_value(1), "The number of episodes a single environment generation seed is reused (= new environment every x episodes).")
            ("seed", po::value<long>()->default_value(-1), "The seed used for the complete run (ignored if -1)")
            ("fix_agent_positions", "If set, the agent starting positions will be fixed across all episodes.")

            // termination options, stop if:
            //   num_games > max_games
            ("max_games", po::value<int>()->default_value(10), "The max. number of generated games (ignored if -1)")
            //   || num_samples >= max_samples (hard cut)
            ("max_samples", po::value<int>()->default_value(-1), "The max. number of logged samples (ignored if -1)")
            //   || num_samples >= targeted_samples (soft cut, episode will still be added as long as num_samples < max_samples)
            ("targeted_samples", po::value<int>()->default_value(-1), "The targeted number of logged samples, fully includes the last episode (ignored if -1). ")

            // log options
            ("log", "If set, generate enough samples to fill a whole dataset (chunk_size * chunk_count samples)")
            ("file_prefix", po::value<std::string>()->default_value("./data"), "Set the filename prefix for the new datasets")
            ("chunk_size", po::value<int>()->default_value(1000), "Max. number of samples in a single file inside the dataset")
            ("chunk_count", po::value<int>()->default_value(100), "Max. number of chunks in a dataset")

            // mcts options
            ("model_dir", po::value<std::string>()->default_value("./model"), "The directory which contains the agent's model(s) for multiple batch sizes")
            ("raw_net_agent", "If set, uses the raw net agent instead of the mcts agent.")
            // TODO: State size should be detected automatically (?)
            ("state_size", po::value<uint>()->default_value(0), "Size of the flattened state of the model (0 for no state)")
            ("simulations", po::value<int>()->default_value(100), "Size of the flattened state of the model (0 for no state)")
            ("movetime", po::value<int>()->default_value(100), "Size of the flattened state of the model (0 for no state)")
            ("planning_agents", po::value<std::string>()->default_value("SimpleUnbiasedAgent"), "Agent type used during planning")
            ("value_version", po::value<uint>()->default_value(1), "1 = considers only win/loss, 2 = considers defeated agents")
    ;

    po::variables_map configVals;
    po::store(po::parse_command_line(argc, argv, configDesc), configVals);
    po::notify(configVals);

    if (configVals.count("help")) {
        std::cout << configDesc << "\n";
        return 1;
    }

    long seed = configVals["seed"].as<long>();
    if(seed == -1)
    {
        seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    }

    // check whether we want to log the games
    std::unique_ptr<FileBasedIPCManager> ipcManager;
    int maxSamples = configVals["max_samples"].as<int>();
    if (configVals.count("log")) {
        ipcManager = std::make_unique<FileBasedIPCManager>(configVals["file_prefix"].as<std::string>(), configVals["chunk_size"].as<int>(), configVals["chunk_count"].as<int>());

        // fill at most one dataset
        int oneDataSet = configVals["chunk_size"].as<int>() * configVals["chunk_count"].as<int>();
        maxSamples = maxSamples == -1 ? oneDataSet : min(maxSamples, oneDataSet);
    }

    RunnerConfig config;
    config.maxEpisodeSteps = 800;
    config.maxEpisodes = configVals["max_games"].as<int>();
    config.targetedLoggedSteps = configVals["targeted_samples"].as<int>();;
    config.maxLoggedSteps = maxSamples;
    config.seed = seed;
    config.envSeed = configVals["env_seed"].as<long>();
    config.envGenSeedEps = configVals["env_gen_seed_eps"].as<long>();
    config.randomAgentPositions = configVals.count("fix_agent_positions") == 0;
    config.printSteps = configVals.count("print") > 0;
    config.printFirstLast = configVals.count("print_first_last") > 0;
    config.ipcManager = ipcManager.get();

    std::string mode = configVals["mode"].as<std::string>();
    if (mode == "ffa_sl") {
        Runner::run_simple_agents(config);
    }
    else if (mode == "ffa_mcts") {
        bool useRawNetAgent = configVals.count("raw_net_agent") > 0;
        std::string modelDir = configVals["model_dir"].as<std::string>();

        PlanningAgentType planningAgentType;
        std::string planningAgentStr = configVals["planning_agents"].as<std::string>();
        if (planningAgentStr == "SimpleUnbiasedAgent")
        {
            planningAgentType = PlanningAgentType::SimpleUnbiasedAgent;
        }
        else if (planningAgentStr == "SimpleAgent")
        {
            planningAgentType = PlanningAgentType::SimpleAgent;
        }
        else if (planningAgentStr == "LazyAgent")
        {
            planningAgentType = PlanningAgentType::LazyAgent;
        }
        else
        {
            std::cerr << "Unknown planning agent type: " << planningAgentStr << std::endl;
            return 1;
        }

        SearchLimits searchLimits;
        searchLimits.simulations = configVals["simulations"].as<int>();
        searchLimits.movetime = configVals["movetime"].as<int>();

        free_for_all_tourney(modelDir, config, useRawNetAgent, configVals["state_size"].as<uint>(), configVals["value_version"].as<uint>(), planningAgentType, searchLimits);
    }
    else {
        std::cerr << "Unknown mode: " << mode << std::endl;
        return 1;
    }

    if(ipcManager.get() != nullptr)
    {
        ipcManager->flush();
    }

    return 0;
}
