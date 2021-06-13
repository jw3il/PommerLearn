#include "pymethods.hpp"
#include "agents.hpp"
#include "crazyara_agent.h"

#include <iostream>
#include "string.h"

CrazyAraAgent* create_crazyara_agent(SearchLimits searchLimits, std::string modelDir, uint stateSize)
{
    StateConstants::init(false);
    StateConstantsPommerman::set_auxiliary_outputs(stateSize);
    
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

std::pair<std::string, std::string> _extract_arg(std::string input, std::string delimiter=":")
{
    auto delimiterPos = input.find(":");
    if(delimiterPos == std::string::npos)
    {
        return std::pair<std::string, std::string>("", input);
    }

    std::string arg = input.substr(0, delimiterPos);
    std::string remainder = input.substr(delimiterPos + 1, input.length() - delimiterPos - 1);
    return std::pair<std::string, std::string>(arg, remainder);
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
        auto tmp = _extract_arg(agentName);
        tmp = _extract_arg(tmp.second);
        std::string modelDir = tmp.first;
        tmp = _extract_arg(tmp.second);
        std::string stateSize = tmp.first;
        tmp = _extract_arg(tmp.second);
        std::string simulations = tmp.first;
        std::string moveTime = tmp.second;
        if(modelDir.empty() || stateSize.empty() || simulations.empty() || moveTime.empty()) 
        {
            std::cout << "Could not parse agent identifier. Specify the agent like CrazyAra:dir:stateSize:simulations:moveTime" << std::endl;
            return nullptr;
        }

        std::cout << "Creating CrazyAraAgent with " << std::endl
                  << "> Model dir: " << modelDir << std::endl
                  << "> State size: " << stateSize << std::endl
                  << "> Simulations: " << simulations << std::endl
                  << "> Movetime: " << moveTime << std::endl;

        SearchLimits searchLimits;
        searchLimits.simulations = std::stoi(simulations);
        searchLimits.movetime = std::stoi(moveTime);
        return create_crazyara_agent(searchLimits, modelDir, std::stoi(stateSize));
    }

    return nullptr;
}