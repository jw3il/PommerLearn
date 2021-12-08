#ifndef POMMERMANCRAZYARAAGENT_H
#define POMMERMANCRAZYARAAGENT_H

#include "agents.hpp"
#include "pommermanstate.h"
#include "agent.h"
#include "log_agent.h"
#include "data_representation.h"
#include "neuralnetapi.h"

/**
 * @brief Agent types that can be used during planning.
 */
enum PlanningAgentType 
{
    SimpleUnbiasedAgent,
    SimpleAgent,
    LazyAgent
};

/**
 * @brief Wrapper for any kind of crazyara::Agent (e.g. RawNetAgent, MCTSAgent) used as a bboard::Agent.
 */
class CrazyAraAgent : public LogAgent
{
private:
    std::unique_ptr<crazyara::Agent> agent;
    std::unique_ptr<NeuralNetAPI> singleNet;
    vector<unique_ptr<NeuralNetAPI>> netBatches;
    PlaySettings playSettings;
    SearchSettings searchSettings;
    SearchLimits searchLimits;

    std::unique_ptr<PommermanState> pommermanState;
    EvalInfo evalInfo;
    std::array<Clonable<bboard::Agent>*, bboard::AGENT_COUNT> planningAgents;

    float planeBuffer[PLANES_TOTAL_FLOATS];
    float policyBuffer[NUM_MOVES];
    float qBuffer[NUM_MOVES];

    bboard::Environment* env = nullptr;
public:
    const bool isRawNetAgent;

public:
    CrazyAraAgent(std::string modelDirectory);
    CrazyAraAgent(std::string modelDirectory, PlaySettings playSettings, SearchSettings searchSettings, SearchLimits searchLimits);

    void init_state(bboard::GameMode gameMode, bboard::ObservationParameters obsParams, bboard::ObservationParameters opponentObsParams, uint8_t valueVersion, PlanningAgentType planningAgentType=PlanningAgentType::SimpleUnbiasedAgent);

    /**
     * @brief Use the true state of the given environment instead of the observation in the act method.
     * Note that the view of this agent is still controlled via the ObservationParameters given in init_state.
     * 
     * @param env The environment. Call with nullptr to remove a previously set environment and use regular observations again.
     */
    void use_environment_state(bboard::Environment* env);

    // helper methods
    static std::unique_ptr<NeuralNetAPI> load_network(const std::string& modelDirectory);
    static vector<unique_ptr<NeuralNetAPI>> load_network_batches(const std::string& modelDirectory, const SearchSettings& searchSettings);
    static SearchSettings get_default_search_settings(const bool selfPlay);

    crazyara::Agent* get_agent();

    // bboard::Agent
    bboard::Move act(const bboard::Observation* obs) override;
    void reset() override;
};

#endif // POMMERMANCRAZYARAAGENT_H
