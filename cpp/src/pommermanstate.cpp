/*
 * @file: pommermanstate.cpp
 * Created on 15.07.2020
 * @author: queensgambit
 */

#include "pommermanstate.h"
#include "data_representation.h"
#include "agents.hpp"


PommermanState::PommermanState(uint agentID, bboard::GameMode gameMode):
    agentID(agentID),
    gameMode(gameMode),
    usePartialObservability(false),
    eventHash(0),
    hasPlanningAgents(false),
    hasBufferedActions(false)
{
    std::fill_n(moves, bboard::AGENT_COUNT, bboard::Move::IDLE);
}

void PommermanState::set_state(const bboard::State* state)
{
    if(this->usePartialObservability)
    {
        // simulate partial observability

        bboard::Observation obs;
        // TODO: Maybe directly use the state object instead of observations?
        bboard::Observation::Get(*state, agentID, this->params, obs);
        set_observation(&obs);
    }
    else
    {
        this->state = *state;
    }

    if (hasPlanningAgents) {
        planning_agents_reset();
    }
}

void PommermanState::set_observation(const bboard::Observation* obs)
{
    // TODO: Merge observations
    obs->ToState(this->state, gameMode);

    if (hasPlanningAgents) {
        planning_agents_reset();
    }
}

void PommermanState::set_partial_observability(const bboard::ObservationParameters* params)
{
    this->usePartialObservability = true;
    this->params = *params;

    if (this->hasPlanningAgents) {
        throw std::runtime_error("The combination of partial observability & planning agents is not implemented yet!");
    }
}

void PommermanState::set_planning_agents(const std::array<Clonable<bboard::Agent>*, bboard::AGENT_COUNT> agents)
{
    if (this->usePartialObservability) {
        throw std::runtime_error("The combination of partial observability & planning agents is not implemented yet!");
    }

    hasPlanningAgents = false;
    for (size_t i = 0; i < agents.size(); i++) {
        // skip own id, as we won't use this agent
        if (i == agentID) {
            continue;
        }

        Clonable<bboard::Agent>* agent = agents[i];
        if (agent != nullptr) {
            // create new clonable agent from this one and set its id
            planningAgents[i] = agent->clone();
            planningAgents[i]->get_obj_ptr()->id = i;
            // we have at least one agent
            hasPlanningAgents = true;
        }
    }
}

void PommermanState::planning_agents_reset()
{
    for (size_t i = 0; i < planningAgents.size(); i++) {
        if (i == agentID) {
            continue;
        }

        Clonable<bboard::Agent>* agent = planningAgents[i].get();
        if (agent != nullptr) {
            agent->get_obj_ptr()->reset();
        }
    }

    hasBufferedActions = false;
}

void PommermanState::planning_agents_act()
{
    // we don't have to act when the actions are already buffered
    if (hasBufferedActions)
        return;

    for (size_t i = 0; i < planningAgents.size(); i++) {
        if (i == agentID) {
            continue;
        }

        Clonable<bboard::Agent>* agent = planningAgents[i].get();
        if (agent != nullptr) {
            moves[i] = agent->get_obj_ptr()->act(&state);
        }
    }

    hasBufferedActions = true;
}


// State methods

std::vector<Action> PommermanState::legal_actions() const
{
    const bboard::AgentInfo& self = state.agents[agentID];
    std::vector<Action> legalActions;

    // it's always possible to idle
    legalActions.push_back(Action(bboard::Move::IDLE));

    // agents can only place bombs when max bomb count is not reached yet and they don't already stand on a bomb
    if (self.bombCount < self.maxBombCount && !state.HasBomb(self.x, self.y)) {
        legalActions.push_back(Action(bboard::Move::BOMB));
    }

    // check if movement is possible
    static const bboard::Move directions[4] = {bboard::Move::UP, bboard::Move::DOWN, bboard::Move::RIGHT, bboard::Move::LEFT};
    for (bboard::Move dir : directions) {
        bboard::Position dest = bboard::util::DesiredPosition(self.x, self.y, dir);
        if (bboard::util::IsOutOfBounds(dest)) {
            continue;
        }

        // check if the item at the destination is passable (and don't walk into flames)
        int destItem = state.items[dest.y][dest.x];
        if (destItem == bboard::Item::PASSAGE || bboard::IS_POWERUP(destItem)
                || (destItem == bboard::Item::BOMB && self.canKick)) {
            legalActions.push_back(Action(dir));
        }
    }

    return legalActions;
}

void PommermanState::set(const std::string &fenStr, bool isChess960, int variant)
{
    // TODO
}

void PommermanState::get_state_planes(bool normalize, float *inputPlanes) const
{
    // TODO
    StateToPlanes(&state, 0, inputPlanes);
}

unsigned int PommermanState::steps_from_null() const
{
    return state.timeStep;
}

bool PommermanState::is_chess960() const
{
    return false;
}

std::string PommermanState::fen() const
{
    return "<fen-placeholder>";
}

bool _attribute_changed(const bboard::AgentInfo& oldInfo, const bboard::AgentInfo& newInfo) {
    return oldInfo.canKick != newInfo.canKick
            || oldInfo.bombCount != newInfo.bombCount
            || oldInfo.bombStrength != newInfo.bombStrength
            || oldInfo.maxBombCount != newInfo.maxBombCount;
}

void PommermanState::do_action(Action action)
{
    bboard::AgentInfo info = state.agents[agentID];
    int bombCount = state.bombs.count;
    int flameCount = state.flames.count;

    moves[agentID] = bboard::Move(action);

    if (hasPlanningAgents) {
        // fill the remaining moves
        planning_agents_act();
        // after this step, our cached actions are invalid
        hasBufferedActions = false;
    }

    // std::cout << "Moves: " << (int)moves[0] << " " << (int)moves[1] << " " << (int)moves[2] << " " << (int)moves[3] << std::endl;
    bboard::Step(&state, moves);

    if (_attribute_changed(info, state.agents[agentID])
            || bombCount != state.bombs.count || flameCount != state.flames.count) {
        eventHash = rand();
    }
}

void PommermanState::undo_action(Action action) {
    // TODO
}

void PommermanState::prepare_action()
{
    // TODO
}


unsigned int PommermanState::number_repetitions() const
{
    return 0;
}

int PommermanState::side_to_move() const
{
    return agentID;
}

Key PommermanState::hash_key() const
{
    const bboard::AgentInfo& self = state.agents[agentID];
    int pos = self.x + self.y * bboard::BOARD_SIZE;
    // pos is in range [0, 120], so we can shift left by 7 (128)
    return ((Key)eventHash << 7) + pos;
}

void PommermanState::flip()
{
    // pass
}

Action PommermanState::uci_to_action(std::string &uciStr) const
{
    // TODO
    return Action(bboard::Move::IDLE);
}

std::string PommermanState::action_to_san(Action action, const std::vector<Action>& legalActions, bool leadsToWin, bool bookMove) const
{
    // TODO: Maybe change to UTF8 symbols later
    switch(bboard::Move(action)) {
    case (bboard::Move::IDLE):
        return "I";
    case (bboard::Move::UP):
        return "U";
    case (bboard::Move::DOWN):
        return "D";
    case (bboard::Move::LEFT):
        return "L";
    case (bboard::Move::RIGHT):
        return "R";
    case (bboard::Move::BOMB):
        return "B";
    default:
        return "?";
    }
}

TerminalType PommermanState::is_terminal(size_t numberLegalMoves, bool inCheck, float& customTerminalValue) const
{
    if(state.finished)
    {
        if(state.agents[agentID].won)
        {
            customTerminalValue = 2.0f;
            return TERMINAL_CUSTOM;
        }
        else
        {
            if(state.isDraw || state.timeStep > 800)
            {
                customTerminalValue = 0.0f;
                return TERMINAL_CUSTOM;
            }
            else
            {
                customTerminalValue = -2.0f;
                return TERMINAL_CUSTOM;
            }
        }
    }

    // state is not finished
    if(state.agents[agentID].dead)
    {
        if (gameMode == bboard::GameMode::FreeForAll) {
            customTerminalValue = -2.0f;
            return TERMINAL_CUSTOM;
        }
        // Partner is still alive
        // TODO: Add evaluation from NN
        customTerminalValue = -0.5f;
        return TERMINAL_CUSTOM;
    }

    return TERMINAL_NONE;
}

Result PommermanState::check_result(bool inCheck) const
{
    // TODO
}


bool PommermanState::gives_check(Action action) const
{
    return false;
}

PommermanState* PommermanState::clone() const
{
    PommermanState* clone = new PommermanState(agentID, gameMode);
    clone->state = state;
    if (hasPlanningAgents) {
        // clone all agents
        for (size_t i = 0; i < planningAgents.size(); i++) {
            auto ptr = planningAgents[i].get();
            if (ptr != nullptr) {
                clone->planningAgents[i] = ptr->clone();
            }
        }

        clone->hasBufferedActions = hasBufferedActions;
        if (hasBufferedActions) {
            std::copy_n(moves, bboard::AGENT_COUNT, clone->moves);
        }

        // we'll also use the agents in the clone
        clone->hasPlanningAgents = true;
    }
    return clone;
}

void PommermanState::print(std::ostream& os) const
{
    // TODO
    os << InitialStateToString(state);
}

Tablebase::WDLScore PommermanState::check_for_tablebase_wdl(Tablebase::ProbeState& result)
{
    result = Tablebase::FAIL;
    return Tablebase::WDLScoreNone;
}
