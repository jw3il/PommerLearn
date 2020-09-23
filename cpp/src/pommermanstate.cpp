/*
 * @file: pommermanstate.cpp
 * Created on 15.07.2020
 * @author: queensgambit
 */

#include "pommermanstate.h"
#include "data_representation.h"
#include "agents.hpp"


PommermanState::PommermanState(int agentID, bboard::GameMode gameMode):
    agentID(agentID),
    gameMode(gameMode),
    plies(0),
    usePartialObservability(false)
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
}

void PommermanState::set_observation(const bboard::Observation* obs)
{
    // TODO: Merge observations
    obs->ToState(this->state, gameMode);
}

void PommermanState::set_partial_observability(const bboard::ObservationParameters* params)
{
    this->usePartialObservability = true;
    this->params = *params;
}

std::vector<Action> PommermanState::legal_actions() const
{
    // TODO: Change depending on the current state
    return {Action(bboard::Move::IDLE),
            Action(bboard::Move::UP),
            Action(bboard::Move::LEFT),
            Action(bboard::Move::DOWN),
            Action(bboard::Move::RIGHT),
            Action(bboard::Move::BOMB)};
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
    return plies;
}

bool PommermanState::is_chess960() const
{
    return false;
}

std::string PommermanState::fen() const
{
    return "<fen-placeholder>";
}

void PommermanState::do_action(Action action)
{    
    moves[agentID] = bboard::Move(action);
    bboard::Step(&state, moves);
}

void PommermanState::undo_action(Action action) {
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
    return 0;
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
    return "";
}

TerminalType PommermanState::is_terminal(size_t numberLegalMoves, bool inCheck, float& customTerminalValue) const
{
    if(state.finished)
    {
        if(state.agents[agentID].won)
        {
            return TERMINAL_WIN;
        }
        else
        {
            if(state.isDraw)
            {
                return TERMINAL_DRAW;
            }
            else
            {
                return TERMINAL_LOSS;
            }
        }
    }

    // state is not finished

    if(state.agents[agentID].dead)
    {
        // TODO: Add custom terminal value
        customTerminalValue = -0.5;
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
