#ifndef POMMERMANCRAZYARAAGENT_H
#define POMMERMANCRAZYARAAGENT_H

#include "agents.hpp"
#include "pommermanstate.h"
#include "agent.h"

/**
 * @brief The PommermanCrazyAraAgent class is a wrapper for any kind of crazyara::Agent (e.g. RawNetAgent, MCTSAgent) to be used as a bboard::Agent.
 */
struct CrazyAraAgent : bboard::Agent
{
private:
    crazyara::Agent* agent;
    PommermanState* pommermanState;
    SearchLimits* searchLimits;
    EvalInfo* evalInfo;

public:
    CrazyAraAgent(crazyara::Agent* agent, PommermanState* pommermanState, SearchLimits* searchLimits, EvalInfo* evalInfo);

    // Agent interface
public:
    bboard::Move act(const bboard::State *state);
};

#endif // POMMERMANCRAZYARAAGENT_H
