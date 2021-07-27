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
    SimpleAgent
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

public:
    const bool isRawNetAgent;

public:
    CrazyAraAgent(std::string modelDirectory);
    CrazyAraAgent(std::string modelDirectory, PlaySettings playSettings, SearchSettings searchSettings, SearchLimits searchLimits);

    void init_state(bboard::GameMode gameMode, bboard::ObservationParameters observationParameters, uint8_t valueVersion, PlanningAgentType planningAgentType=PlanningAgentType::SimpleUnbiasedAgent);

    // helper methods
    static std::unique_ptr<NeuralNetAPI> load_network(const std::string& modelDirectory);
    static vector<unique_ptr<NeuralNetAPI>> load_network_batches(const std::string& modelDirectory, const SearchSettings& searchSettings);
    static SearchSettings get_default_search_settings(const bool selfPlay);

    // bboard::Agent
    bboard::Move act(const bboard::State *state) override;
    void reset() override;
};

#endif // POMMERMANCRAZYARAAGENT_H
