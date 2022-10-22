#ifndef POMMERCRAZYARAAGENT_H
#define POMMERCRAZYARAAGENT_H

#include "agents.hpp"
#include "pommermanstate.h"
#include "agent.h"
#include "log_agent.h"
#include "data_representation.h"
#include "neuralnetapi.h"
#include "safe_ptr_queue.h"
#include "rawnetagent.h"
#include "mctsagent.h"

#ifdef TENSORRT
#include "nn/tensorrtapi.h"
#elif defined (TORCH)
#include "nn/torchapi.h"
#endif

/**
 * @brief Agent types that can be used during planning.
 */
enum PlanningAgentType 
{
    None,
    SimpleUnbiasedAgent,
    SimpleAgent,
    LazyAgent,
    RawNetworkAgent  // "RawNetworkAgent" instead of "RawNetAgent" is used to avoid naming conflict
};

/**
 * @brief Base class for wrapping CrazyAra agents as Pommerman agents.
 */
class CrazyAraAgent : public LogAgent, public Clonable<bboard::Agent>
{
protected:
    std::unique_ptr<PommermanState> pommermanState;

    SearchLimits searchLimits;
    PlaySettings playSettings;
    EvalInfo evalInfo;

    std::unique_ptr<float[]> planeBuffer;
    std::unique_ptr<float[]> policyBuffer;
    std::unique_ptr<float[]> qBuffer;

    bboard::Environment* env = nullptr;

public:
    /**
     * @brief Initializes the Pommermanstate.
     * 
     * @param gameMode The gamemode
     * @param obsParams How this agent observes the state
     * @param opponentObsParams How opponents observe the state (only relevant for MCTS)
     * @param valueVersion The value version (only relevant for MCTS)
     */
    void init_state(bboard::GameMode gameMode, bboard::ObservationParameters obsParams, bboard::ObservationParameters opponentObsParams, uint8_t valueVersion=1);

    /**
     * @brief Use the true state of the given environment instead of the observation in the act method.
     * Note that the view of this agent is still controlled via the ObservationParameters given in init_state.
     * 
     * @param env The environment. Call with nullptr to remove a previously set environment and use regular observations again.
     */
    void use_environment_state(bboard::Environment* env);

    /**
     * @brief Loads a single network
     * 
     * @param modelDirectory The model directory
     * @return A NeuralNetAPI instance for the model
     */
    static std::unique_ptr<NeuralNetAPI> load_network(const std::string& modelDirectory, const int deviceID);

    /**
     * @brief Get a pointer to the crazyara agent that is used in act (warning: only well-defined within act)
     * 
     * @return The acting agent
     */
    virtual crazyara::Agent* get_acting_agent() = 0;

    /**
     * @brief Get a pointer to the crazyara agent that is used in act (warning: only well-defined within act)
     * 
     * @return The network of the acting agent
     */
    virtual NeuralNetAPI* get_acting_net() = 0;

    /**
     * @brief Return whether the used model is stateful.
     * 
     * @return true iff the model is stateful
     */
    virtual bool has_stateful_model() = 0;

    // bboard::Agent
    bboard::Move act(const bboard::Observation* obs) override;
    void reset() override;

    virtual ~CrazyAraAgent() {}

    /**
     * @brief set_pommerman_state Sets the current pommerman state object according to the environment or given observation.
     * @param obs Given observation
     */
    void set_pommerman_state(const bboard::Observation *obs);

    /**
     * @brief get_pommerman_state gets the associated pommerman state.
     */
    const PommermanState* get_pommerman_state() const;

    /**
     * @brief print_pv_line Prints the PV line of the mcts agent. This function is supposed to be called after the search.
     * @param mctsAgent MCTS Agent object
     * @param obs Current observation
     */
    void print_pv_line(MCTSAgent* mctsAgent, const bboard::Observation *obs);

    /**
     * @brief add_results_to_buffer Stores the search result in a buffer for later look-up
     * @param net Neural net object
     * @param bestAction Best found action after the search
     */
    void add_results_to_buffer(const NeuralNetAPI* net, bboard::Move bestAction);
};

struct RawNetAgentContainer {
    std::unique_ptr<RawNetAgent> agent;
    std::unique_ptr<NeuralNetAPI> net;
    std::unique_ptr<PlaySettings> playSettings;
};

/**
 * @brief Pommerman agent using a thread-safe queue of RawNetAgents (queue can be shared across threads).
 */
class RawCrazyAraAgent : public CrazyAraAgent
{
private:
    std::unique_ptr<RawNetAgentContainer> currentAgent;
    std::shared_ptr<SafePtrQueue<RawNetAgentContainer>> rawNetAgentQueue;

public:
    RawCrazyAraAgent(std::shared_ptr<SafePtrQueue<RawNetAgentContainer>> rawNetAgentQueue);
    RawCrazyAraAgent(const std::string& modelDirectory, const int deviceID);

    bboard::Move act(const bboard::Observation* obs) override;

    /**
     * @brief Create a queue of raw net agents.
     * 
     * @param modelDirectory The model directory
     * @param count The number of agents in the queue
     * @return new queue containing size RawNetAgentContainers
     */
    static std::unique_ptr<SafePtrQueue<RawNetAgentContainer>> load_raw_net_agent_queue(const std::string& modelDirectory, int count, const int deviceID);

    // CrazyAraAgent
    bool has_stateful_model() override;
    crazyara::Agent* get_acting_agent() override;
    NeuralNetAPI* get_acting_net() override;

    // Clonable
    bboard::Agent* get() override;

    /**
     * @brief Clone this agent by creating a new agent that shares it's network queue with this agent.
     * @return the cloned agent
     */
    std::unique_ptr<Clonable<bboard::Agent>> clone() override;
};

/**
 * @brief Pommerman agent using a MCTSAgent.
 */
class MCTSCrazyAraAgent : public CrazyAraAgent 
{
private:
    std::unique_ptr<MCTSAgent> agent;
    std::unique_ptr<NeuralNetAPI> singleNet;
    vector<std::unique_ptr<NeuralNetAPI>> netBatches;

    SearchSettings searchSettings;
    const std::string modelDirectory;
    const int deviceID;
    PlanningAgentType planningAgentType = PlanningAgentType::None;
    int switchDepth = -1;

public:
    MCTSCrazyAraAgent(const std::string& modelDirectory, const int deviceID, PlaySettings playSettings, SearchSettings searchSettings, SearchLimits searchLimits);

    static vector<unique_ptr<NeuralNetAPI>> load_network_batches(const std::string& modelDirectory, const int deviceID, const SearchSettings& searchSettings);
    static SearchSettings get_default_search_settings(const bool selfPlay);

    /**
     * @brief Initializes the planning agents (note that the state has to be initialized first).
     *
     * @param planningAgentType The (primary) planning agent type
     * @param switchDepth Switch from planningAgentType to SimpleUnbiasedAgents at switchDepth if >= 0
     */
    void init_planning_agents(PlanningAgentType planningAgentType, int switchDepth=-1);

    // CrazyAraAgent
    bool has_stateful_model() override;
    crazyara::Agent* get_acting_agent() override;
    NeuralNetAPI* get_acting_net() override;

    // Clonable
    bboard::Agent* get() override;

    /**
     * @brief Clone this agent by creating a new agent with the same configuration.
     * @return the cloned agent
     */
    std::unique_ptr<Clonable<bboard::Agent>> clone() override;
};

#endif // POMMERCRAZYARAAGENT_H
