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


class PommermanState : public State
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
    std::vector<Action> legal_actions() const override;
    void set(const std::string &fenStr, bool isChess960, int variant) override;
    void get_state_planes(bool normalize, float *inputPlanes) const override;
    unsigned int steps_from_null() const override;
    bool is_chess960() const override;
    std::string fen() const override;
    void do_action(Action action) override;
    void undo_action(Action action) override;
    unsigned int number_repetitions() const override;
    int side_to_move() const override;
    Key hash_key() const override;
    void flip() override;
    Action uci_to_action(std::string &uciStr) const override;
    std::string action_to_san(Action action, const std::vector<Action>& legalActions, bool leadsToWin, bool bookMove) const override;
    TerminalType is_terminal(size_t numberLegalMoves, bool inCheck, float& customTerminalValue) const override;
    Result check_result(bool inCheck) const override;
    bool gives_check(Action action) const override;
    void print(std::ostream& os) const override;
    Tablebase::WDLScore check_for_tablebase_wdl(Tablebase::ProbeState& result) override;
    PommermanState* clone() const override;
};

#endif // POMMERMANSTATE_H
