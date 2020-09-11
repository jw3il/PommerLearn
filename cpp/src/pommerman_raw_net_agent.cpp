#include "pommerman_raw_net_agent.h"

PommermanRawNetAgent::PommermanRawNetAgent(RawNetAgent *rawNetAgent, PommermanState *pommermanState):
    rawNetAgent(rawNetAgent),
    pommermanState(pommermanState)
{
}

bboard::Move PommermanRawNetAgent::act(const bboard::State *state)
{
    pommermanState->set_state(state);
    SearchLimits searchLimits;
    EvalInfo evalInfo;
    rawNetAgent->set_search_settings(pommermanState, &searchLimits, &evalInfo);
    rawNetAgent->perform_action();
    return bboard::Move(rawNetAgent->get_best_action());
}
