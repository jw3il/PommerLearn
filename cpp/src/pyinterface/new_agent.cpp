#include "pymethods.hpp"
#include "agents.hpp"
#include "crazyara_agent.h"

#include <iostream>
#include "string.h"

CrazyAraAgent* create_crazyara_agent(SearchLimits searchLimits, std::string modelDir)
{
    SearchSettings searchSettings = CrazyAraAgent::get_default_search_settings(false);
    PlaySettings playSettings;
    CrazyAraAgent* crazyAraAgent = new CrazyAraAgent(modelDir, playSettings, searchSettings, searchLimits);

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
    else if(agentName.find("CrazyAra") == 0)
    {
        if(agentName.find(":") == std::string::npos) 
        {
            std::cout << "Missing model directory! Specify it like CrazyAra:dir" << std::endl;
            return nullptr;
        }

        int startIndex = agentName.find(":") + 1;
        std::string modelDir = agentName.substr(startIndex, agentName.length() - startIndex);

        SearchLimits searchLimits;
        if(agentName.find("CrazyAra100"))
        {
            searchLimits.simulations = 100;
            searchLimits.movetime = 100;
        }
        else if(agentName.find("CrazyAra500"))
        {
            searchLimits.simulations = 500;
            searchLimits.movetime = 100;
        }
        else
        {
            std::cout << "Unknown CrazyAraAgent" << std::endl;
            return nullptr;
        }

        return create_crazyara_agent(searchLimits, modelDir);
    }

    return nullptr;
}