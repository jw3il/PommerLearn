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

Item mapPyToBoard(int py)
{
    switch (py)
    {
        case 0: return Item::PASSAGE;
        case 1: return Item::RIGID;
        case 2: return Item::WOOD;
        case 3: return Item::BOMB;
        case 4: return Item::FLAME;
        case 5: return Item::FOG;
        case 6: return Item::EXTRABOMB;
        case 7: return Item::INCRRANGE;
        case 8: return Item::KICK;
        case 9: return Item::AGENTDUMMY;
        case 10: return Item::AGENT0;
        case 11: return Item::AGENT1;
        case 12: return Item::AGENT2;
        case 13: return Item::AGENT3;
        default: throw std::runtime_error("Unknown map item " + std::to_string(py));
    }
}

Direction mapPyToDir(int py)
{
    switch (py)
    {
        case 0: return Direction::IDLE;
        case 1: return Direction::UP;
        case 2: return Direction::DOWN;
        case 3: return Direction::LEFT;
        case 4: return  Direction::RIGHT;
        default: throw std::runtime_error("Unknown direction " + std::to_string(py));
    }
}

void PyToBoard(const nlohmann::json& pyBoard, State& state)
{
    for(int y = 0; y < BOARD_SIZE; y++)
    {
        for(int x = 0; x < BOARD_SIZE; x++)
        {
            state.board[y][x] = mapPyToBoard(pyBoard[y][x].get<int>());
        }
    }
}

void PyToAgentInfo(const nlohmann::json& pyInfo, AgentInfo& info)
{
    // attributes: agent_id, ammo, blast_strength, can_kick, is_alive, position (tuple)

    // agent_id is defined by the info index
    // info.team and info.bombCount must be set outside this function

    // Agent positions are stored (row, column)
    info.x = pyInfo["position"][1];
    info.y = pyInfo["position"][0];

    info.dead = !pyInfo["is_alive"];

    info.canKick = pyInfo["can_kick"];
    info.maxBombCount = pyInfo["ammo"];
    info.bombStrength = pyInfo["blast_strength"];
}

void PyToBomb(const nlohmann::json& pyBomb, Bomb& bomb)
{
    // attributes: bomber_id, moving_direction, position (tuple), life, blast_strength

    SetBombID(bomb, pyBomb["bomber_id"]);
    const nlohmann::json& pos = pyBomb["position"];
    // Bomb positions are stored (column, row)
    SetBombPosition(bomb, pos[0], pos[1]);
    SetBombStrength(bomb, pyBomb["blast_strength"]);

    nlohmann::json movingDir = pyBomb["moving_direction"];
    if(movingDir.is_null())
    {
        SetBombDirection(bomb, Direction::IDLE);
    }
    else
    {
        SetBombDirection(bomb, mapPyToDir(movingDir));
    }

    SetBombMovedFlag(bomb, false);
    // TODO: Check for inconsistency
    SetBombTime(bomb, pyBomb["life"]);
}

void PyToFlame(const nlohmann::json& pyFlame, Flame& flame)
{
    // attributes: position (tuple), life

    const nlohmann::json& pos = pyFlame["position"];
    // Flame positions are stored (row, column)
    flame.position.x = pos[1];
    flame.position.y = pos[0];
    flame.timeLeft = pyFlame["life"];
}

bool sortByTimeLeft(const Flame &lhs, const Flame &rhs)
{
    return lhs.timeLeft < rhs.timeLeft;
}

template <int count>
void printFlames(FixedQueue<Flame, count>& flames)
{
    for(int i = 0; i < flames.count; i++)
    {
        std::cout << i << " (" << flames[i].timeLeft << ", " << flames[i].position.x << ", " << flames[i].position.y << ") ";
    }

    std::cout << std::endl;
}

int OptimizeFlameQueue(State& s)
{
    // sort flames
    std::sort(s.flames.queue, s.flames.queue + s.flames.count, sortByTimeLeft);

    // modify timeLeft (additive)
    int timeLeft = 0;
    for(int i = 0; i < s.flames.count; i++)
    {
        Flame& f = s.flames[i];
        int oldVal = f.timeLeft;
        f.timeLeft -= timeLeft;
        timeLeft = oldVal;

        // set flame ids to allow for faster lookup
        s.board[f.position.y][f.position.x] += (i << 3);
    }

    // return total time left
    return timeLeft;
}

void PyStringToState(const std::string& string, State& state, GameMode gameMode)
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

        // assign teams
        switch (gameMode)
        {
            case GameMode::FreeForAll:
                info.team = 0;
                break;
            case GameMode::TwoTeams:
                info.team = (i % 2 == 0) ? 1 : 2;
                break;
        }

        if(!info.dead && state.board[info.y][info.x] < Item::AGENT0)
        {
            throw std::runtime_error("Expected agent, got " + std::to_string(state.board[info.y][info.x]));
        }
    }

    // set bombs
    const nlohmann::json& pyBombs = pyState["bombs"];
    for(uint i = 0; i < pyBombs.size(); i++)
    {
        Bomb bomb;
        PyToBomb(pyBombs[i], bomb);
        state.bombs.AddElem(bomb);

        // increment agent bomb count
        state.agents[BMB_ID(bomb)].bombCount++;
    }

    // set flames
    const nlohmann::json& pyFlames = pyState["flames"];
    for(uint i = 0; i < pyFlames.size(); i++)
    {
        Flame flame;
        PyToFlame(pyFlames[i], flame);
        state.flames.AddElem(flame);

        if(!IS_FLAME(state.board[flame.position.y][flame.position.x]))
        {
            throw std::runtime_error("Invalid flame @ " + std::to_string(flame.position.x) + ", " + std::to_string(flame.position.y));
        }
    }

    // set items
    const nlohmann::json& pyItems = pyState["items"];
    for(uint i = 0; i < pyItems.size(); i++)
    {
        // attributes: tuple(position (tuple), type)
        const nlohmann::json& pyItem = pyItems[i];

        const nlohmann::json& pos = pyItem[0];
        const Item type = mapPyToBoard(pyItem[1]);

        // Item position is (row, column)
        int& boardItem = state.board[pos[0].get<int>()][pos[1].get<int>()];
        switch (boardItem) {
            case Item::PASSAGE:
                boardItem = type;
                break;
            case Item::WOOD:
            case Item::FLAME:
                boardItem += State::ItemFlag(type);
                break;
            default:
                throw std::runtime_error("Powerup at board item " + std::to_string(boardItem));
        }
    }

    // optimize flames for faster steps
    state.currentFlameTime = OptimizeFlameQueue(state);
}

State PyStringToState(const std::string& string, GameMode gameMode)
{
    State state;
    PyStringToState(string, state, gameMode);
    return state;
}

void PyStringToObservation(const std::string& string, Observation& obs)
{
    nlohmann::json jstate = nlohmann::json::parse(string);

    // attributes:
    // - game_type (int), game_env (string), step_count (int)
    // - alive (list with ids), enemies (list with ids),
    // - position (int pair), blast_strength (int), can_kick (bool), teammate (list with ids), ammo (int),
    // - board (int matrix), bomb_blast_strength (float matrix), bomb_life (float matrix), bomb_moving_direction (float matrix), flame_life (float matrix)

}

Observation PyStringToObservation(const std::string& string)
{
    Observation obs;
    PyStringToObservation(string, obs);
    return obs;
}

#endif // POMCPP_UTIL_H
