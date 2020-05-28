#include "log_agent.h"

LogAgent::LogAgent(int maxEpisodeLength) {
    this->agent = nullptr;
    this->step = 0;
    this->id = -1;

    this->actionBuffer = new bboard::Move[maxEpisodeLength];
    this->stateBuffer = new bboard::State[maxEpisodeLength];
}

bboard::Move LogAgent::act(const bboard::State* state) {
    bboard::Move move = this->agent == nullptr ? bboard::Move::IDLE : this->agent->act(state);

    // log the (state, action) pair
    this->stateBuffer[step] = *state;
    this->actionBuffer[step] = move;

    this->step++;

    return move;
}

void LogAgent::reset(bboard::Agent* agent) {
    this->step = 0;
    this->agent = agent;
}

void LogAgent::deleteAgent() {
    if (this->agent != nullptr)
        delete this->agent;

    this->step = 0;
    this->agent = nullptr;
}
