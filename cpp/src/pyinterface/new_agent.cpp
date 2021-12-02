#include "pymethods.hpp"
#include "agents.hpp"
#include "crazyara_agent.h"

#include <iostream>
#include "string.h"

std::unique_ptr<CrazyAraAgent> create_crazyara_agent(std::string modelDir, uint stateSize, bool rawNetAgent, SearchLimits searchLimits=SearchLimits())
{
    StateConstants::init(false);
    StateConstantsPommerman::set_auxiliary_outputs(stateSize);
    
    std::unique_ptr<CrazyAraAgent> crazyAraAgent;
    if(rawNetAgent)
    {
        crazyAraAgent = std::make_unique<CrazyAraAgent>(modelDir);
    }
    else
    {
        SearchSettings searchSettings = CrazyAraAgent::get_default_search_settings(false);
        PlaySettings playSettings;
        crazyAraAgent = std::make_unique<CrazyAraAgent>(modelDir, playSettings, searchSettings, searchLimits);
    }

    // partial observability
    bboard::ObservationParameters obsParams;
    obsParams.agentPartialMapView = false;
    obsParams.agentInfoVisibility = bboard::AgentInfoVisibility::All;
    obsParams.exposePowerUps = false;

    uint valueVersion = 2;
    crazyAraAgent->init_state(bboard::GameMode::FreeForAll, obsParams, obsParams, valueVersion);

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

std::unique_ptr<bboard::Agent> PyInterface::new_agent(std::string agentName, long seed)
{
    if(agentName == "SimpleAgent")
    {
        return std::make_unique<agents::SimpleAgent>(seed);
    }
    else if(agentName == "SimpleUnbiasedAgent")
    {
        return std::make_unique<agents::SimpleUnbiasedAgent>(seed);
    }
    else if(agentName.find("RawNetAgent") == 0)
    {
        auto tmp = _extract_arg(agentName);
        tmp = _extract_arg(tmp.second);
        std::string modelDir = tmp.first;
        std::string stateSize = tmp.second;
        if(modelDir.empty() || stateSize.empty()) 
        {
            std::cout << "Could not parse agent identifier. Specify the agent like RawNet:dir:stateSize" << std::endl;
            return nullptr;
        }

        std::cout << "Creating RawNetAgent with " << std::endl
            << "> Model dir: " << modelDir << std::endl
            << "> State size: " << stateSize << std::endl;

        return create_crazyara_agent(modelDir, std::stoi(stateSize), true);
    }
    else if(agentName.find("CrazyAraAgent") == 0)
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
        return create_crazyara_agent(modelDir, std::stoi(stateSize), false, searchLimits=searchLimits);
    }

    return nullptr;
}
