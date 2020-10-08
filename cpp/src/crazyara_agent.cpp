#include "crazyara_agent.h"

CrazyAraAgent::CrazyAraAgent(crazyara::Agent* agent, PommermanState* pommermanState, SearchLimits* searchLimits, EvalInfo* evalInfo):
    agent(agent),
    pommermanState(pommermanState),
    searchLimits(searchLimits),
    evalInfo(evalInfo)
{

}

bboard::Move CrazyAraAgent::act(const bboard::State *state)
{
    pommermanState->set_state(state);
    agent->set_search_settings(pommermanState, searchLimits, evalInfo);
    agent->perform_action();

    for(uint i = 0; i < evalInfo->legalMoves.size(); i++)
    {
        std::cout << StateConstantsPommerman::action_to_uci(evalInfo->legalMoves.at(i), false) << " "; // (" << evalInfo->movesToMate.at(i) << ") ";
    }
    std::cout << std::endl;

    return bboard::Move(agent->get_best_action());
}

void CrazyAraAgent::reset() {

}
