#include "depth_switch_agent.h"

DepthSwitchAgent::DepthSwitchAgent(std::unique_ptr<Clonable<bboard::Agent>> agent0, int agent0Depth, long seed): agent(std::move(agent0)), seed(seed), agent0Depth(agent0Depth) {}

std::unique_ptr<Clonable<bboard::Agent>> DepthSwitchAgent::create_agent_1()
{
    return std::unique_ptr<Clonable<bboard::Agent>>(
        new CopyClonable<bboard::Agent, agents::SimpleUnbiasedAgent>(agents::SimpleUnbiasedAgent(seed))
    );
}

bboard::Move DepthSwitchAgent::act(const bboard::Observation* obs)
{
    if (timeStepsAfterReset == -1) {
        timeStepsAfterReset = obs->timeStep;
    }

    int depth = obs->timeStep - timeStepsAfterReset;

    if (depth > agent0Depth) {
        agent = create_agent_1();
    }

    return agent->get()->act(obs);
}

void DepthSwitchAgent::reset()
{
    timeStepsAfterReset = -1;
}

bboard::Agent* DepthSwitchAgent::get()
{
    return this;
}

std::unique_ptr<Clonable<bboard::Agent>> DepthSwitchAgent::clone()
{
    if (!isAgent1) {
        // just clone the underlying agent0 and remember that we want to switch later
        auto myClone = std::make_unique<DepthSwitchAgent>(agent->clone(), agent0Depth, seed);
        myClone->timeStepsAfterReset = timeStepsAfterReset;
        myClone->isAgent1 = isAgent1;
        return myClone;
    }

    // if we are already agent 1, then we will never switch back again.
    // => we can actually drop the wrapper and only return a clone of the underlying agent
    return agent->clone();
}