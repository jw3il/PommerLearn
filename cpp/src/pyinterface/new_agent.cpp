#include "pymethods.hpp"
#include "agents.hpp"
#include "crazyara_agent.h"

#include <iostream>
#include "string.h"

CrazyAraAgent* create_crazyara_agent(SearchLimits searchLimits)
{
    SearchSettings searchSettings = CrazyAraAgent::get_default_search_settings(false);
    PlaySettings playSettings;
    CrazyAraAgent* crazyAraAgent = new CrazyAraAgent("./model/torch_cpu/", playSettings, searchSettings, searchLimits);

    // partial observability
    bboard::ObservationParameters obsParams;
    obsParams.agentPartialMapView = false;
    obsParams.agentInfoVisibility = bboard::AgentInfoVisibility::All;
    obsParams.exposePowerUps = false;

    crazyAraAgent->init_state(bboard::GameMode::FreeForAll, obsParams);

    return crazyAraAgent;
}

bboard::Agent* PyInterface::new_agent(std::string agentName, long seed)
{
    if(agentName == "SimpleAgent")
    {
        return new agents::SimpleAgent(seed);
    }
    else if(agentName == "SimpleUnbiasedAgent")
    {
        return new agents::SimpleUnbiasedAgent(seed);
    }
    else if(agentName == "CrazyAra100")
    {
        SearchLimits searchLimits;
        searchLimits.simulations = 100;
        searchLimits.movetime = 100;
        return create_crazyara_agent(searchLimits);
    }
    else if(agentName == "CrazyAra500")
    {
        SearchLimits searchLimits;
        searchLimits.simulations = 500;
        searchLimits.movetime = 100;
        return create_crazyara_agent(searchLimits);
    }

    return nullptr;
}