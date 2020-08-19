/*
 * @file: pommermanstate.h
 * Created on 15.07.2020
 * @author: queensgambit
 *
 * PommermanState implements the State interface for the Pommerman C++ environment.
 */

#ifndef POMMERMANSTATE_H
#define POMMERMANSTATE_H

#include "state.h"
#include "bboard.hpp"


class PommermanState : public State<PommermanState>
{
public:
    PommermanState();
    ~PommermanState();
    bboard::Environment env;
    const unsigned int numberAgents = 4;
    bboard::Move* agentActions;
    size_t agentToMove;
    unsigned int plies;

    // State interface
public:
    std::vector<Action> legal_actions() const;
    void set(const std::string &fenStr, bool isChess960, int variant);
    void get_state_planes(bool normalize, float *inputPlanes) const;
    unsigned int steps_from_null() const;
    bool is_chess960() const;
    std::string fen() const;
    void do_action(Action action);
    void undo_action(Action action);
    unsigned int number_repetitions() const;
    int side_to_move() const;
    Key hash_key() const;
    void flip();
    Action uci_to_action(std::string &uciStr) const;
    std::string action_to_san(Action action, const std::vector<Action>& legalActions, bool leadsToWin, bool bookMove) const;
    TerminalType is_terminal(size_t numberLegalMoves, bool inCheck, float& customTerminalValue) const;
    Result check_result(bool inCheck) const;
    bool gives_check(Action action) const;
    void print(std::ostream& os) const;
    std::unique_ptr<PommermanState> clone() const;
};

#endif // POMMERMANSTATE_H
