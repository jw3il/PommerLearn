#include "crazyara_agent.h"
#include "depth_switch_agent.h"

MCTSCrazyAraAgent::MCTSCrazyAraAgent(const std::string& modelDirectory, const int deviceID, PlaySettings playSettings, SearchSettings searchSettings, SearchLimits searchLimits):
    modelDirectory(modelDirectory), deviceID(deviceID)
{
    this->playSettings = playSettings;
    this->searchSettings = searchSettings;
    this->searchLimits = searchLimits;
    this->singleNet = load_network(modelDirectory, deviceID);
    this->netBatches = load_network_batches(modelDirectory, deviceID, searchSettings);
    agent = std::make_unique<MCTSAgent>(this->singleNet.get(), this->netBatches, &this->searchSettings, &this->playSettings);
}

vector<unique_ptr<NeuralNetAPI>> MCTSCrazyAraAgent::load_network_batches(const string& modelDirectory, const int deviceID, const SearchSettings& searchSettings)
{
    vector<unique_ptr<NeuralNetAPI>> netBatches;
#ifdef MXNET
    #ifdef TENSORRT
        const bool useTensorRT = bool(Options["Use_TensorRT"]);
    #else
        const bool useTensorRT = false;
    #endif
#endif
    int First_Device_ID = deviceID;
    int Last_Device_ID = deviceID;
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
    netBatches[0]->validate_neural_network();
    return netBatches;
}

SearchSettings MCTSCrazyAraAgent::get_default_search_settings(const bool selfPlay)
{
    SearchSettings searchSettings;

    searchSettings.virtualLoss = 1;
    searchSettings.batchSize = 8;
    searchSettings.threads = 4;
    searchSettings.useMCGS = false;
    searchSettings.multiPV = 1;
    searchSettings.nodePolicyTemperature = 1.0f;
    if (selfPlay)
    {
        searchSettings.dirichletEpsilon = 0.25f;
    }
    else
    {
        searchSettings.dirichletEpsilon = 0;
    }
    searchSettings.dirichletAlpha = 0.2f;
    searchSettings.epsilonGreedyCounter = 0;
    searchSettings.epsilonChecksCounter = 0;
    searchSettings.qVetoDelta = 0.4;
    searchSettings.qValueWeight = 1.0f;
    searchSettings.reuseTree = false;
    searchSettings.mctsSolver = false;

    return searchSettings;
}

void MCTSCrazyAraAgent::init_planning_agents(PlanningAgentType planningAgentType, int switchDepth)
{
    this->planningAgentType = planningAgentType;
    this->switchDepth = switchDepth;

    if (planningAgentType == PlanningAgentType::None) {
        return;
    }

    if (!pommermanState) {
        throw std::runtime_error("The state has to be created before planning agents can be assigned to it.");
    }

    // create raw net agent queue that is shared by all planning raw net agents
    std::shared_ptr<SafePtrQueue<RawNetAgentContainer>> rawNetAgentQueue;
    if (planningAgentType == PlanningAgentType::RawNetworkAgent) {
        if (modelDirectory.empty()) {
            throw std::runtime_error("Cannot use planningAgentType RawNetworkAgent with empty model directory.");
        }

        rawNetAgentQueue = RawCrazyAraAgent::load_raw_net_agent_queue(modelDirectory, deviceID, searchSettings.threads);
    }

    for (int i = 0; i < bboard::AGENT_COUNT; i++) {
        std::unique_ptr<Clonable<bboard::Agent>> agent;

        // other agents used for planning
        switch (planningAgentType)
        {
        case PlanningAgentType::SimpleUnbiasedAgent:
        {
            agent = std::unique_ptr<Clonable<bboard::Agent>>(
                new CopyClonable<bboard::Agent, agents::SimpleUnbiasedAgent>(agents::SimpleUnbiasedAgent(rand()))
            );
            break;
        }
        case PlanningAgentType::SimpleAgent:
        {
            agent = std::unique_ptr<Clonable<bboard::Agent>>(
                new CopyClonable<bboard::Agent, agents::SimpleAgent>(agents::SimpleAgent(rand()))
            );
            break;
        }
        case PlanningAgentType::LazyAgent:
        {
            agent = std::unique_ptr<Clonable<bboard::Agent>>(
                new CopyClonable<bboard::Agent, agents::LazyAgent>(agents::LazyAgent())
            );
            break;
        }
        case PlanningAgentType::RawNetworkAgent:
        {
            std::unique_ptr<RawCrazyAraAgent> crazyAraAgent = std::make_unique<RawCrazyAraAgent>(rawNetAgentQueue);
            crazyAraAgent->id = i;
            crazyAraAgent->init_state(pommermanState->gameMode, pommermanState->opponentObsParams, pommermanState->opponentObsParams, pommermanState->valueVersion);
            agent = std::move(crazyAraAgent);
            break;
        }
        default:
            break;
        }

        if (switchDepth >= 0) {
            agent = std::make_unique<DepthSwitchAgent>(std::move(agent), switchDepth, rand());
        }

        pommermanState->set_planning_agent(std::move(agent), i);
    }
}

bool MCTSCrazyAraAgent::has_stateful_model()
{
    return singleNet->has_auxiliary_outputs();
}

crazyara::Agent* MCTSCrazyAraAgent::get_acting_agent()
{
    return agent.get();
}

NeuralNetAPI* MCTSCrazyAraAgent::get_acting_net()
{
    return singleNet.get();
}

bboard::Agent* MCTSCrazyAraAgent::get()
{
    return this;
}

std::unique_ptr<Clonable<bboard::Agent>> MCTSCrazyAraAgent::clone()
{
    if (!pommermanState.get()) {
        throw std::runtime_error("Cannot clone agent with uninitialized state!");
    }

    std::unique_ptr<MCTSCrazyAraAgent> clonedAgent = std::make_unique<MCTSCrazyAraAgent>(modelDirectory, deviceID, playSettings, searchSettings, searchLimits);

    clonedAgent->id = id;
    clonedAgent->pommermanState = std::unique_ptr<PommermanState>(pommermanState->clone());
    // we have to reset the planning agents to disconnect the clones if they use shared networks
    clonedAgent->init_planning_agents(planningAgentType, switchDepth);

    return clonedAgent;
}
