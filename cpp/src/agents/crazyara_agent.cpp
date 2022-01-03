#include "crazyara_agent.h"
#include <limits>

#include "agents/rawnetagent.h"
#include "agents/mctsagent.h"

void CrazyAraAgent::init_state(bboard::GameMode gameMode, bboard::ObservationParameters observationParameters, uint8_t valueVersion, PlanningAgentType planningAgentType)
{
    pommermanState = std::make_unique<PommermanState>(gameMode, has_stateful_model(), 800, valueVersion);
    pommermanState->set_partial_observability(observationParameters);
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

bboard::Move CrazyAraAgent::act(const bboard::Observation *obs)
{
    pommermanState->set_observation(obs);

    crazyara::Agent* agent = get_acting_agent();
    NeuralNetAPI* net = get_acting_net();

    agent->set_search_settings(pommermanState.get(), &searchLimits, &evalInfo);
    agent->perform_action();

    bboard::Move bestAction = bboard::Move(agent->get_best_action());

    #ifndef DISABLE_UCI_INFO
        MCTSAgent* mctsAgent = dynamic_cast<MCTSAgent*>(agent);
        if (mctsAgent != nullptr) {
            mctsAgent->print_root_node();
        }
    #endif

    #ifdef CRAZYARA_AGENT_PV
        MCTSAgent* mctsAgent = dynamic_cast<MCTSAgent*>(agent);
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

    if (has_buffer()) {
        if (!planeBuffer) {
            // create buffers on the fly
            planeBuffer = std::unique_ptr<float[]>(new float[PLANES_TOTAL_FLOATS]);
            policyBuffer = std::unique_ptr<float[]>(new float[NUM_MOVES]);
            qBuffer = std::unique_ptr<float[]>(new float[NUM_MOVES]);
        }
        pommermanState->get_state_planes(true, planeBuffer.get(), net->get_version());
        _get_policy(&evalInfo, policyBuffer.get());
        _get_q(&evalInfo, qBuffer.get());

        sampleBuffer->addSample(planeBuffer.get(), bestAction, policyBuffer.get(), evalInfo.bestMoveQ.at(0), qBuffer.get());
    }
    return bestAction;
}

void CrazyAraAgent::reset() {
    // update ID of the agent of MCTS states
    pommermanState->set_agent_id(id);
}

std::unique_ptr<NeuralNetAPI> CrazyAraAgent::load_network(const std::string& modelDirectory)
{
#ifdef TENSORRT
    return std::make_unique<TensorrtAPI>(0, 1, modelDirectory, "float32");
#elif defined (TORCH)
    return std::make_unique<TorchAPI>("cpu", 0, 1, modelDirectory);
#endif
}
