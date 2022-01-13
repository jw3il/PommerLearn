#ifndef DEPTH_SWITCH_AGENT_H
#define DEPTH_SWITCH_AGENT_H

#include "agents.hpp"
#include "clonable.h"

/**
 * @brief An agent that switches from one agent to another after a given depth. 
 * This base class switches from any agent to a SimpleUnbiasedAgent.
 * Note that this switch is not undone after the reset.
 */
class DepthSwitchAgent : public bboard::Agent, public Clonable<bboard::Agent>
{
private:
    std::unique_ptr<Clonable<bboard::Agent>> agent;
    const int agent0Depth;
    int timeStepsAfterReset = -1;
    bool isAgent1 = false;

    const long seed;

public:
    /**
     * @brief Create a new DepthSwitchAgent.
     * 
     * @param agent0 the agent that is used until depth agent0Depth
     * @param agent0Depth the depth until which agent0 is used 
     * @param seed the seed for the agent that is created for the steps beyond agent0Depth 
     */
    DepthSwitchAgent(std::unique_ptr<Clonable<bboard::Agent>> agent0, int agent0Depth, long seed);

    /*
        Note: switching between agent0 and agent1 is currently implemented by creating agent1 on the
        fly with create_agent_1 instead of providing an additional argument for the constructor. 
        
        This way, we can avoid unnecessary clones of agent1 before agent0Depth is reached (but it is less
        flexible). If you want to switch between many different agent1 classes and extending this
        class becomes a hassle, you should consider adding an additional argument for agent1 instead.
    */

    /**
     * @brief Create an instance of the agent that will be used after agent0.
     * 
     * @return a new agent
     */
    std::unique_ptr<Clonable<bboard::Agent>> create_agent_1();

    bboard::Move act(const bboard::Observation* obs) override;
    void reset() override;

    // Clonable
    bboard::Agent* get() override;
    std::unique_ptr<Clonable<bboard::Agent>> clone() override;
};

#endif // DEPTH_SWITCH_AGENT_H