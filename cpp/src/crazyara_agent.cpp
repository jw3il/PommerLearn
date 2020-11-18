#include "crazyara_agent.h"
#include <limits>

CrazyAraAgent::CrazyAraAgent(crazyara::Agent* agent, PommermanState* pommermanState, SearchLimits* searchLimits, EvalInfo* evalInfo):
    agent(agent),
    pommermanState(pommermanState),
    searchLimits(searchLimits),
    evalInfo(evalInfo)
{

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

bboard::Move CrazyAraAgent::act(const bboard::State *state)
{
    pommermanState->set_state(state);
    agent->set_search_settings(pommermanState, searchLimits, evalInfo);
    agent->perform_action();

    bboard::Move bestAction = bboard::Move(agent->get_best_action());

    if (has_buffer()) {
        pommermanState->get_state_planes(true, planeBuffer);
        _get_policy(evalInfo, policyBuffer);
        _get_q(evalInfo, qBuffer);

        sampleBuffer->addSample(planeBuffer, bestAction, policyBuffer, evalInfo->bestMoveQ.at(0), qBuffer);
    }

    return bestAction;
}

void CrazyAraAgent::reset() {

}
