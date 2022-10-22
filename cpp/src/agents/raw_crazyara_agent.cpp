#include "crazyara_agent.h"

RawCrazyAraAgent::RawCrazyAraAgent(std::shared_ptr<SafePtrQueue<RawNetAgentContainer>> rawNetAgentQueue)
{
    // share the network
    this->rawNetAgentQueue = rawNetAgentQueue;
}

RawCrazyAraAgent::RawCrazyAraAgent(const std::string& modelDirectory, const int deviceID): 
    RawCrazyAraAgent(load_raw_net_agent_queue(modelDirectory, 1, deviceID)) {}

bboard::Move RawCrazyAraAgent::act(const bboard::Observation *obs)
{
    currentAgent = rawNetAgentQueue->dequeue();
    bboard::Move move = CrazyAraAgent::act(obs);
    rawNetAgentQueue->enqueue(std::move(currentAgent));
    return move;
}

std::unique_ptr<SafePtrQueue<RawNetAgentContainer>> RawCrazyAraAgent::load_raw_net_agent_queue(const std::string& modelDirectory, int count, int deviceID)
{
    auto netQueue = std::make_unique<SafePtrQueue<RawNetAgentContainer>>();

    for (int i = 0; i < count; i++) {
        auto container = std::make_unique<RawNetAgentContainer>();
        // agent uses default playsettings, are not used anyway
        container->playSettings = std::make_unique<PlaySettings>();
        // load the network
        container->net = load_network(modelDirectory, deviceID);
        // .. and create a new agent (this creates a new NeuralNetAPIUser and allocates VRAM)
        container->agent = std::make_unique<RawNetAgent>(container->net.get(), container->playSettings.get(), false);

        // add the container to the queue -> can be used by threads
        netQueue->enqueue(std::move(container));
    }

    return netQueue;
}

bool RawCrazyAraAgent::has_stateful_model()
{
    auto rawNetAgent = rawNetAgentQueue->dequeue();
    bool statefulModel = rawNetAgent->net->has_auxiliary_outputs();
    rawNetAgentQueue->enqueue(std::move(rawNetAgent));
    return statefulModel;
}

bboard::Agent* RawCrazyAraAgent::get()
{
    return this;
}

std::unique_ptr<Clonable<bboard::Agent>> RawCrazyAraAgent::clone()
{
    if (!pommermanState.get()) {
        throw std::runtime_error("Cannot clone agent with uninitialized state!");
    }

    std::unique_ptr<RawCrazyAraAgent> clonedAgent = std::make_unique<RawCrazyAraAgent>(rawNetAgentQueue);

    clonedAgent->id = id;
    clonedAgent->pommermanState = std::unique_ptr<PommermanState>(pommermanState->clone());

    return clonedAgent;
}

crazyara::Agent* RawCrazyAraAgent::get_acting_agent()
{
    if (currentAgent) {
        return currentAgent->agent.get();
    }

    return nullptr;
}

NeuralNetAPI* RawCrazyAraAgent::get_acting_net()
{
    if (currentAgent) {
        return currentAgent->net.get();
    }

    return nullptr;
}
