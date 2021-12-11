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
    None,
    SimpleUnbiasedAgent,
    SimpleAgent,
    LazyAgent,
    RawNetworkAgent  // "RawNetworkAgent" instead of "RawNetAgent" is used to avoid naming conflict
};

/**
 * @brief Wrapper for any kind of crazyara::Agent (e.g. RawNetAgent, MCTSAgent) used as a bboard::Agent.
 */
class CrazyAraAgent : public LogAgent, public Clonable<bboard::Agent>
{
private:
    std::shared_ptr<crazyara::Agent> agent;
    std::shared_ptr<NeuralNetAPI> singleNet;
    vector<std::shared_ptr<NeuralNetAPI>> netBatches;
    PlaySettings playSettings;
    SearchSettings searchSettings;
    SearchLimits searchLimits;

    std::unique_ptr<PommermanState> pommermanState;
    EvalInfo evalInfo;

    std::unique_ptr<float[]> planeBuffer;
    std::unique_ptr<float[]> policyBuffer;
    std::unique_ptr<float[]> qBuffer;
    bool isRawNetAgent;

public:
    // raw net agent constructors
    CrazyAraAgent(const std::shared_ptr<NeuralNetAPI> singleNet);
    CrazyAraAgent(const std::shared_ptr<crazyara::Agent> agent);
    CrazyAraAgent(const std::string& modelDirectory);
    
    // mcts agent constructors
    CrazyAraAgent(std::shared_ptr<NeuralNetAPI> singleNet, vector<std::shared_ptr<NeuralNetAPI>> netBatches, PlaySettings playSettings, SearchSettings searchSettings, SearchLimits searchLimits);
    CrazyAraAgent(const std::string& modelDirectory, PlaySettings playSettings, SearchSettings searchSettings, SearchLimits searchLimits);

    void init_state(bboard::GameMode gameMode, bboard::ObservationParameters observationParameters, uint8_t valueVersion, PlanningAgentType planningAgentType=PlanningAgentType::None);

    // helper methods
    static std::unique_ptr<NeuralNetAPI> load_network(const std::string& modelDirectory);
    static vector<shared_ptr<NeuralNetAPI>> load_network_batches(const std::string& modelDirectory, const SearchSettings& searchSettings);
    static SearchSettings get_default_search_settings(const bool selfPlay);

    crazyara::Agent* get_agent();

    // bboard::Agent
    bboard::Move act(const bboard::Observation* obs) override;
    void reset() override;

    // Clonable
    bboard::Agent* get() override;
    std::unique_ptr<Clonable<bboard::Agent>> clone() override;
};

#endif // POMMERMANCRAZYARAAGENT_H
