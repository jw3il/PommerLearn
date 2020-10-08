#ifndef POMMERMANCRAZYARAAGENT_H
#define POMMERMANCRAZYARAAGENT_H

#include "agents.hpp"
#include "pommermanstate.h"
#include "agent.h"
#include "log_agent.h"

/**
 * @brief The PommermanCrazyAraAgent class is a wrapper for any kind of crazyara::Agent (e.g. RawNetAgent, MCTSAgent) to be used as a bboard::Agent.
 */
struct CrazyAraAgent : public bboard::Agent
{
private:
    crazyara::Agent* agent;
    PommermanState* pommermanState;
    SearchLimits* searchLimits;
    EvalInfo* evalInfo;

public:
    CrazyAraAgent(crazyara::Agent* agent, PommermanState* pommermanState, SearchLimits* searchLimits, EvalInfo* evalInfo);

    // bboard::Agent
    bboard::Move act(const bboard::State *state) override;
    void reset() override;
};

#endif // POMMERMANCRAZYARAAGENT_H
