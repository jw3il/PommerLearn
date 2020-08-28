#ifndef POMCPP_UTIL_H
#define POMCPP_UTIL_H

#include "bboard.hpp"
#include "step_utility.hpp"
#include "nlohmann/json.hpp"
#include <unordered_set>

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
            state.items[y][x] = mapPyToBoard(pyBoard[y][x].get<int>());
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

    info.dead = !pyInfo.value("is_alive", true);

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

template <int count>
void printFlames(FixedQueue<Flame, count>& flames)
{
    for(int i = 0; i < flames.count; i++)
    {
        std::cout << i << " (" << flames[i].timeLeft << ", " << flames[i].position.x << ", " << flames[i].position.y << ") ";
    }

    std::cout << std::endl;
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

        if(!info.dead && state.items[info.y][info.x] < Item::AGENT0)
        {
            throw std::runtime_error("Expected agent, got " + std::to_string(state.items[info.y][info.x]));
        }
    }

    // set bombs
    const nlohmann::json& pyBombs = pyState["bombs"];
    state.bombs.count = 0;
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
    state.flames.count = 0;
    for(uint i = 0; i < pyFlames.size(); i++)
    {
        Flame flame;
        PyToFlame(pyFlames[i], flame);
        state.flames.AddElem(flame);

        if(!IS_FLAME(state.items[flame.position.y][flame.position.x]))
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
        int& boardItem = state.items[pos[0].get<int>()][pos[1].get<int>()];
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
    state.currentFlameTime = util::OptimizeFlameQueue(state);
}

State PyStringToState(const std::string& string, GameMode gameMode)
{
    State state;
    PyStringToState(string, state, gameMode);
    return state;
}

void PyStringToObservation(const std::string& string, int agentId, Observation& obs)
{
    nlohmann::json pyState = nlohmann::json::parse(string);

    // attributes:
    // - game_type (int), game_env (string), step_count (int)
    // - alive (list with ids), enemies (list with ids),
    // - position (int pair), blast_strength (int), can_kick (bool), teammate (list with ids), ammo (int),
    // - board (int matrix), bomb_blast_strength (float matrix), bomb_life (float matrix), bomb_moving_direction (float matrix), flame_life (float matrix)
    const nlohmann::json& alive = pyState["alive"];
    std::fill_n(obs.isAlive, AGENT_COUNT, false);
    for(uint i = 0; i < alive.size(); i++)
    {
        obs.isAlive[alive[i].get<int>()] = true;
    }

    const nlohmann::json& enemies = pyState["enemies"];
    std::fill_n(obs.isEnemy, AGENT_COUNT, false);
    for(uint i = 0; i < enemies.size(); i++)
    {
        obs.isEnemy[enemies[i].get<int>()] = true;
    }

    // we only observe ourself
    std::fill_n(obs.agentIDMapping, AGENT_COUNT, -1);
    obs.agentIDMapping[agentId] = 0;
    obs.agentInfos.count = 1;
    AgentInfo& info = obs.agentInfos[0];

    // set agent info
    PyToAgentInfo(pyState, info);
    info.dead = obs.isAlive[agentId];

    // set board

    // try to reconstruct own bombs from given observation
    std::unordered_set<Position> ownBombs;
    for(int i = 0; i < obs.bombs.count; i++)
    {
        Bomb& b = obs.bombs[i];
        if(BMB_ID(b) == agentId)
        {
            ownBombs.insert({BMB_POS_X(b), BMB_POS_Y(b)});
        }
    }

    obs.bombs.count = 0;
    obs.flames.count = 0;
    const nlohmann::json& pyBoard = pyState["board"];
    for(int y = 0; y < BOARD_SIZE; y++)
    {
        for(int x = 0; x < BOARD_SIZE; x++)
        {
            Item item = mapPyToBoard(pyBoard[y][x].get<int>());
            obs.items[y][x] = item;

            switch (item)
            {
                case Item::FLAME:
                {
                    Flame& f = obs.flames.NextPos();
                    f.position.x = x;
                    f.position.y = y;
                    f.timeLeft = (int)pyState["flame_life"][y][x].get<float>();
                    obs.flames.count++;
                    break;
                }
                case Item::BOMB:
                {
                    Bomb& b = obs.bombs.NextPos();
                    SetBombPosition(b, x, y);

                    // is that necessary?
                    SetBombMovedFlag(b, false);

                    // WARNING: Bomber Id is not not known! This means based on a single observation,
                    // we do not know when our ammo fills back up in the future.

                    // because of this, we try to reconstruct our own bombs based on the last observation
                    // idea: when performing an action, remember agent ids in the last observation struct

                    // TODO: This simple approximation breaks when bombs move. Maybe one could include
                    // the moving direction to reconstruct agent ids of moving bombs.
                    if(ownBombs.find({x, y}) != ownBombs.end())
                    {
                        SetBombID(b, agentId);
                    }
                    else
                    {
                        // illegal agent id
                        SetBombID(b, AGENT_COUNT);
                    }

                    int blastStrength = (int)pyState["bomb_blast_strength"][y][x].get<float>();
                    SetBombStrength(b, blastStrength);

                    Direction direction = mapPyToDir((int)pyState["bomb_moving_direction"][y][x].get<float>());
                    SetBombDirection(b, direction);

                    int life = (int)pyState["bomb_life"][y][x].get<float>();
                    SetBombTime(b, life);

                    obs.bombs.count++;
                    break;
                }
                default: break;
            }
        }
    }

    obs.currentFlameTime = util::OptimizeFlameQueue(obs);
}

Observation PyStringToObservation(const std::string& string, int agentId)
{
    Observation obs;
    PyStringToObservation(string, agentId, obs);
    return obs;
}

#endif // POMCPP_UTIL_H
