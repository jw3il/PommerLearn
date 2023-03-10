#include "depth_switch_agent.h"

DepthSwitchAgent::DepthSwitchAgent(std::unique_ptr<Clonable<bboard::Agent>> agent0, int switchDepth, long seed): agent(std::move(agent0)),  switchDepth(switchDepth), seed(seed) {}

std::unique_ptr<Clonable<bboard::Agent>> DepthSwitchAgent::create_agent_1()
{
    return std::unique_ptr<Clonable<bboard::Agent>>(
        new CopyClonable<bboard::Agent, agents::SimpleUnbiasedAgent>(agents::SimpleUnbiasedAgent(seed))
    );
}

void _reset_agent_with_id(bboard::Agent* agent, int id)
{
    if (agent) {
        agent->id = id;
        agent->reset();
    }
}

bboard::Move DepthSwitchAgent::act(const bboard::Observation* obs)
{
    if (timeStepsAfterReset == -1) {
        timeStepsAfterReset = obs->timeStep;
    }

    int depth = obs->timeStep - timeStepsAfterReset;

    if (!isAgent1 && depth >= switchDepth) {
        agent = create_agent_1();
        // we cannot transfer state between agents => reset agent within the episode
        _reset_agent_with_id(agent->get(), id);
        isAgent1 = true;
    }

    return agent->get()->act(obs);
}

void DepthSwitchAgent::reset()
{
    timeStepsAfterReset = -1;
    _reset_agent_with_id(agent->get(), id);
}

bboard::Agent* DepthSwitchAgent::get()
{
    return this;
}

std::unique_ptr<Clonable<bboard::Agent>> DepthSwitchAgent::clone()
{
    if (!isAgent1) {
        // just clone the underlying agent0 and remember that we want to switch later
        auto myClone = std::make_unique<DepthSwitchAgent>(agent->clone(), switchDepth, seed);
        myClone->id = id;
        myClone->timeStepsAfterReset = timeStepsAfterReset;
        myClone->isAgent1 = isAgent1;
        return myClone;
    }

    // if we are already agent 1, then we will never switch back again.
    // => we can actually drop the wrapper and only return a clone of the underlying agent
    return agent->clone();
}
