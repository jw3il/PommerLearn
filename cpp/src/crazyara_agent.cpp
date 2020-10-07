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
    return bboard::Move(agent->get_best_action());
}
