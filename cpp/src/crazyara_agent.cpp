#include "crazyara_agent.h"
#include <limits>

#ifdef TENSORRT
#include "nn/tensorrtapi.h"
#elif defined (TORCH)
#include "nn/torchapi.h"
#endif

#include "agents/rawnetagent.h"
#include "agents/mctsagent.h"

CrazyAraAgent::CrazyAraAgent(std::shared_ptr<SafePtrQueue<RawNetAgentContainer>> rawNetAgentQueue):
    isRawNetAgent(true)
{
    // share the network
    this->rawNetAgentQueue = rawNetAgentQueue;
}

CrazyAraAgent::CrazyAraAgent(const std::string& modelDirectory): 
    CrazyAraAgent(load_raw_net_agent_queue(modelDirectory, 1)) 
{
    this->modelDirectory = modelDirectory;
}

CrazyAraAgent::CrazyAraAgent(std::unique_ptr<NeuralNetAPI> singleNet, std::vector<std::unique_ptr<NeuralNetAPI>> netBatches, PlaySettings playSettings, SearchSettings searchSettings, SearchLimits searchLimits):
    playSettings(playSettings),
    searchSettings(searchSettings),
    searchLimits(searchLimits),
    isRawNetAgent(false)
{
    this->singleNet = std::move(singleNet);
    this->netBatches = std::move(netBatches);
    agent = std::make_unique<MCTSAgent>(this->singleNet.get(), this->netBatches, &this->searchSettings, &this->playSettings);
}

CrazyAraAgent::CrazyAraAgent(const std::string& modelDirectory, PlaySettings playSettings, SearchSettings searchSettings, SearchLimits searchLimits):
    CrazyAraAgent(load_network(modelDirectory), load_network_batches(modelDirectory, searchSettings), playSettings, searchSettings, searchLimits)
{
    this->modelDirectory = modelDirectory;
}

bboard::Agent* CrazyAraAgent::get()
{
    return this;
}

std::unique_ptr<Clonable<bboard::Agent>> CrazyAraAgent::clone()
{
    if (!pommermanState.get()) {
        throw std::runtime_error("Cannot clone agent with uninitialized state!");
    }

    std::unique_ptr<CrazyAraAgent> clonedCrazyAraAgent;
    
    if (isRawNetAgent) {
        clonedCrazyAraAgent = std::make_unique<CrazyAraAgent>(rawNetAgentQueue);
    }
    else {
        throw std::runtime_error("Cloning is only implemented for RawNetAgents.");
    }

    clonedCrazyAraAgent->id = id;
    clonedCrazyAraAgent->pommermanState = std::unique_ptr<PommermanState>(pommermanState->clone());

    return clonedCrazyAraAgent;
}

void CrazyAraAgent::init_state(bboard::GameMode gameMode, bboard::ObservationParameters observationParameters, uint8_t valueVersion, PlanningAgentType planningAgentType)
{
    bool statefulModel;
    // assuming that the model is stateful when we have auxiliary outputs
    if (isRawNetAgent) {
        auto rawNetAgent = rawNetAgentQueue->dequeue();
        statefulModel = rawNetAgent->net->has_auxiliary_outputs();
        rawNetAgentQueue->enqueue(std::move(rawNetAgent));
    }
    else {
        statefulModel = singleNet->has_auxiliary_outputs();
    }
    
    pommermanState = std::make_unique<PommermanState>(gameMode, statefulModel, 800, valueVersion);
    pommermanState->set_partial_observability(observationParameters);

    if (!this->isRawNetAgent) {
        if (planningAgentType == PlanningAgentType::RawNetworkAgent) {
            if (modelDirectory.empty()) {
                throw std::runtime_error("Cannot use planningAgentType RawNetworkAgent with empty model directory.");
            }

            rawNetAgentQueue = load_raw_net_agent_queue(modelDirectory, searchSettings.threads);
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
                std::unique_ptr<CrazyAraAgent> crazyAraAgent = std::make_unique<CrazyAraAgent>(rawNetAgentQueue);
                crazyAraAgent->id = i;
                crazyAraAgent->init_state(gameMode, observationParameters, valueVersion);
                agent = std::move(crazyAraAgent);
                break;
            }
            default:
                break;
            }

            pommermanState->set_planning_agent(std::move(agent), i);
        }
    }
}

void _get_policy(EvalInfo* info, float* policyProbs) {
    std::fill_n(policyProbs, NUM_MOVES, 0);
    for (size_t i = 0; i < info->legalMoves.size(); i++) {
        Action a = info->legalMoves.at(i);
        float prob = info->policyProbSmall.at(i);

        size_t index = StateConstantsPommerman::action_to_index(a);
        policyProbs[index] = prob;
    }
}

void _get_q(EvalInfo* info, float* q) {
    std::fill_n(q, NUM_MOVES, std::numeric_limits<float>::quiet_NaN());
    for (size_t i = 0; i < info->legalMoves.size(); i++) {
        Action a = info->legalMoves.at(i);
        float qVal = info->qValues.at(i);

        size_t index = StateConstantsPommerman::action_to_index(a);
        q[index] = qVal;
    }
}

void _print_arr(float* arr, int count) {
    if (count <= 0) {
        return;
    }
    for (size_t i = 0; i < count - 1; i++) {
        std::cout << arr[i] << ", ";
    }
    std::cout << arr[count - 1] << std::endl;
}

void _print_q(EvalInfo* info) {
    std::cout << "Q values: ";
    for (size_t i = 0; i < info->legalMoves.size(); i++) {
        Action a = info->legalMoves.at(i);
        float qVal = info->qValues.at(i);

        std::cout << StateConstantsPommerman::action_to_uci(a, false) << ": " << qVal;
        if (i < info->legalMoves.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << std::endl;
}

bboard::Move CrazyAraAgent::act(const bboard::Observation *obs)
{
    pommermanState->set_observation(obs);

    crazyara::Agent* agentPtr = agent.get();

    std::unique_ptr<RawNetAgentContainer> agentContainer;
    if (isRawNetAgent) {
        // get agent from queue
        agentContainer = rawNetAgentQueue->dequeue();
        agentPtr = agentContainer->agent.get();
    }

    agentPtr->set_search_settings(pommermanState.get(), &searchLimits, &evalInfo);
    agentPtr->perform_action();

#ifndef DISABLE_UCI_INFO
    MCTSAgent* mctsAgent = dynamic_cast<MCTSAgent*>(agentPtr.get());
    if (mctsAgent != nullptr) {
        mctsAgent->print_root_node();
    }
#endif

#ifdef CRAZYARA_AGENT_PV
    MCTSAgent* mctsAgent = dynamic_cast<MCTSAgent*>(agentPtr.get());
    if (mctsAgent != nullptr) {
        if (evalInfo.pv.size() > 0) {
            std::cout << "BEGIN ========== Principal Variation" << std::endl;
            pommermanState->set_state(state);
            std::cout << "(" << pommermanState->state.timeStep << "): Initial state" << std::endl;
            bboard::PrintState(&pommermanState->state, false);
            for (Action a : evalInfo.pv[0]) {
                pommermanState->do_action(a);
                std::cout << "(" << pommermanState->state.timeStep << "): after " << StateConstantsPommerman::action_to_uci(a, false) << std::endl;
                bboard::PrintState(&pommermanState->state, false);
            }
            std::cout << "END ========== " << std::endl;
        }
    }
#endif

    bboard::Move bestAction = bboard::Move(agentPtr->get_best_action());

    if (has_buffer()) {
        if (!planeBuffer) {
            // create buffers on the fly
            planeBuffer = std::unique_ptr<float[]>(new float[PLANES_TOTAL_FLOATS]);
            policyBuffer = std::unique_ptr<float[]>(new float[NUM_MOVES]);
            qBuffer = std::unique_ptr<float[]>(new float[NUM_MOVES]);
        }
        pommermanState->get_state_planes(true, planeBuffer.get(), singleNet->get_version());
        _get_policy(&evalInfo, policyBuffer.get());
        _get_q(&evalInfo, qBuffer.get());

        sampleBuffer->addSample(planeBuffer.get(), bestAction, policyBuffer.get(), evalInfo.bestMoveQ.at(0), qBuffer.get());
    }

    if (isRawNetAgent) {
        // put agent back into queue
        rawNetAgentQueue->enqueue(std::move(agentContainer));
    }

    return bestAction;
}

void CrazyAraAgent::reset() {
    // update ID of the agent of MCTS states
    pommermanState->set_agent_id(id);
}

crazyara::Agent* CrazyAraAgent::get_agent()
{
    return agent.get();
}

std::unique_ptr<NeuralNetAPI> CrazyAraAgent::load_network(const std::string& modelDirectory)
{
#ifdef TENSORRT
    return std::make_unique<TensorrtAPI>(0, 1, modelDirectory, "float32");
#elif defined (TORCH)
    return std::make_unique<TorchAPI>("cpu", 0, 1, modelDirectory);
#endif
}

std::unique_ptr<SafePtrQueue<RawNetAgentContainer>> CrazyAraAgent::load_raw_net_agent_queue(const std::string& modelDirectory, int size)
{
    auto netQueue = std::make_unique<SafePtrQueue<RawNetAgentContainer>>();

    for (int i = 0; i < size; i++) {
        auto container = std::make_unique<RawNetAgentContainer>();
        // agent uses default playsettings, are not used anyway
        container->playSettings = std::make_unique<PlaySettings>();
        // load the network
        container->net = load_network(modelDirectory);
        // .. and create a new agent (this creates a new NeuralNetAPIUser and allocates VRAM)
        container->agent = std::make_unique<RawNetAgent>(container->net.get(), container->playSettings.get(), false);

        // add the container to the queue -> can be used by threads
        netQueue->enqueue(std::move(container));
    }

    return netQueue;
}

vector<unique_ptr<NeuralNetAPI>> CrazyAraAgent::load_network_batches(const string& modelDirectory, const SearchSettings& searchSettings)
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
    netBatches[0]->validate_neural_network();
    return netBatches;
}

SearchSettings CrazyAraAgent::get_default_search_settings(const bool selfPlay)
{
    SearchSettings searchSettings;

    searchSettings.virtualLoss = 1;
    searchSettings.batchSize = 8;
    searchSettings.threads = 2;
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
