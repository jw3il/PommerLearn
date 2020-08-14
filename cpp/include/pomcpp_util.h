#ifndef POMCPP_UTIL_H
#define POMCPP_UTIL_H

#include "bboard.hpp"
#include "nlohmann/json.hpp"

using namespace bboard;

template <typename T>
inline void check_key_value(const nlohmann::json& j, const std::string& key, const T& val)
{
    if(val != j[key])
    {
        std::cout << "Full json object: " << j << std::endl;
        throw std::runtime_error("Incorrect value for " + key + "! Expected " + std::to_string(val) + ", got " + j[key].dump() + ".");
    }
}

const std::map<int, Item> mapPyToBoard
{
    {0, Item::PASSAGE},
    {1, Item::RIGID},
    {2, Item::WOOD},
    {3, Item::BOMB},
    {4, Item::FLAMES},
    {5, Item::FOG},
    {6, Item::EXTRABOMB},
    {7, Item::INCRRANGE},
    {8, Item::KICK},
    {9, Item::AGENTDUMMY},
    {10, Item::AGENT0},
    {11, Item::AGENT1},
    {12, Item::AGENT2},
    {13, Item::AGENT3},
};

void PyToBoard(const nlohmann::json& pyBoard, State& state)
{
    for(int y = 0; y < BOARD_SIZE; y++)
    {
        for(int x = 0; x < BOARD_SIZE; x++)
        {
            state.board[y][x] = mapPyToBoard.at(pyBoard[y][x].get<int>());
        }
    }
}

void PyToAgentInfo(const nlohmann::json& pyInfo, AgentInfo& info)
{
    // attributes: agent_id, ammo, blast_strength, can_kick, is_alive, position (tuple)

    info.x = pyInfo["position"][0];
    info.y = pyInfo["position"][1];

    info.dead = !pyInfo["is_alive"];

    info.canKick = pyInfo["can_kick"];
    info.maxBombCount = pyInfo["ammo"];
    info.bombStrength = pyInfo["blast_strength"];

    // TODO: info.team, info.bombCount is missing!
}

void PyToBomb(const nlohmann::json& pyBomb, Bomb& bomb)
{
    // attributes: bomber_id, moving_direction, position (tuple), life, blast_strength

    SetBombID(bomb, pyBomb["bomber_id"]);
    const nlohmann::json& pos = pyBomb["position"];
    SetBombPosition(bomb, pos[0], pos[1]);
    SetBombStrength(bomb, pyBomb["blast_strength"]);
    // Warning: This assumes that bboard::Direction == direction
    SetBombDirection(bomb, pyBomb["moving_direction"]);
    SetBombMovedFlag(bomb, false);
    // TODO: Check for inconsistency
    SetBombTime(bomb, pyBomb["life"]);
}

void PyToFlame(const nlohmann::json& pyFlame, Flame& flame)
{
    // attributes: position (tuple), life

    const nlohmann::json& pos = pyFlame["position"];
    flame.position.x = pos[0];
    flame.position.y = pos[1];
    // TODO: Check for inconsistency
    flame.timeLeft = pyFlame["life"];

    // the python env spawns flame objects in every cell
    flame.strength = 1;
}

void PyStringToState(const std::string& string, State& state)
{
    // attributes: board_size, step_count, board, agents, bombs, flames, items, intended_actions
    const nlohmann::json pyState = nlohmann::json::parse(string);

    check_key_value(pyState, "board_size", BOARD_SIZE);

    state.timeStep = pyState["step_count"];

    // set board
    PyToBoard(pyState["board"], state);

    // set agents
    for(uint i = 0; i < bboard::AGENT_COUNT; i++)
    {
        AgentInfo& info = state.agents[i];
        const nlohmann::json& pyInfo = pyState["agents"][i];

        check_key_value(pyInfo, "agent_id", i);

        PyToAgentInfo(pyInfo, info);
    }

    // set bombs
    const nlohmann::json& pyBombs = pyState["bombs"];
    for(uint i = 0; i < pyBombs.size(); i++)
    {
        Bomb bomb;
        PyToBomb(pyBombs[i], bomb);
        state.bombs.AddElem(bomb);
    }

    // set flames
    const nlohmann::json& pyFlames = pyState["flames"];
    for(uint i = 0; i < pyFlames.size(); i++)
    {
        Flame flame;
        PyToFlame(pyFlames[i], flame);
        state.flames.AddElem(flame);

        // TODO: Adjust flame ids on the board!
    }

    // set items
    const nlohmann::json& pyItems = pyState["items"];
    for(uint i = 0; i < pyItems.size(); i++)
    {
        // attributes: tuple(position (tuple), type)
        const nlohmann::json& pyItem = pyItems[i];

        const nlohmann::json& pos = pyItem[0];
        const Item type = mapPyToBoard.at(pyItem[1]);

        int& boardItem = state.board[pos[1].get<int>()][pos[0].get<int>()];
        switch (boardItem) {
            case Item::PASSAGE:
                boardItem = type;
                break;
            case Item::WOOD:
            case Item::FLAMES:
                boardItem += State::ItemFlag(type);
                break;
            default:
                throw std::runtime_error("Powerup at board item" + std::to_string(boardItem));
        }
    }

    // TODO: info.team, info.bombCount
}

State PyStringToState(const std::string& string)
{
    State state;
    PyStringToState(string, state);
    return state;
}

void PyStringToObservation(const std::string& string, Observation& obs)
{
    nlohmann::json jstate = nlohmann::json::parse(string);
    // TODO
}

Observation PyStringToObservation(const std::string& string)
{
    Observation obs;
    PyStringToObservation(string, obs);
    return obs;
}

#endif // POMCPP_UTIL_H
