#ifndef POMMERMANRAWNETAGENT_H
#define POMMERMANRAWNETAGENT_H

#include "agents.hpp"
#include "pommermanstate.h"
#include "rawnetagent.h"

/**
 * @brief The PommermanRawNetAgent struct is a wrapper for connecting CrazyAra's RawNetAgent to bboard::State
 */
struct PommermanRawNetAgent : bboard::Agent {

    RawNetAgent* rawNetAgent;
    PommermanState* pommermanState;
    // Agent interface
public:
    PommermanRawNetAgent(RawNetAgent* rawNetAgent, PommermanState* pommermanState);
    bboard::Move act(const bboard::State *state) override;
};
#
#endif // POMMERMANRAWNETAGENT_H
