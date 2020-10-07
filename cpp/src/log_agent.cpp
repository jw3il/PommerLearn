#include "log_agent.h"
#include "data_representation.h"

LogAgent::LogAgent(int maxEpisodeLength) : sampleBuffer(maxEpisodeLength) {
    this->agent = nullptr;
    this->step = 0;
    this->id = -1;

    this->planeBuffer = new float[GetObsSize(1)];
}

LogAgent::~LogAgent()
{
    delete[] this->planeBuffer;
}

bboard::Move LogAgent::act(const bboard::State* state) {
    bboard::Move move = this->agent == nullptr ? bboard::Move::IDLE : this->agent->act(state);

    // log the (state, action) pair
    StateToPlanes(state, id, this->planeBuffer);
    this->sampleBuffer.addSample(this->planeBuffer, move);

    this->step++;

    return move;
}

void LogAgent::reset(bboard::Agent* agent) {
    this->step = 0;
    this->sampleBuffer.clear();
    this->agent = agent;
    this->agent->id = this->id;
}

void LogAgent::deleteAgent() {
    if (this->agent != nullptr)
        delete this->agent;

    this->step = 0;
    this->agent = nullptr;
}
