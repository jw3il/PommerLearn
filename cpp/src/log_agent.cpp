#include "log_agent.h"
#include "data_representation.h"

// SampleCollector

void SampleCollector::create_buffer(int maxEpisodeLength) {
    sampleBuffer = std::make_unique<SampleBuffer>(maxEpisodeLength);
}

SampleBuffer* SampleCollector::get_buffer() {
    return sampleBuffer.get();
}

bool SampleCollector::has_buffer() const {
    return sampleBuffer.get() != nullptr;
}

// LogAgent

int LogAgent::get_buffer_agent_id() {
    return this->id;
}

// WrappedLogAgent

WrappedLogAgent::WrappedLogAgent() {
    this->agent = nullptr;
    this->id = -1;

    this->planeBuffer = new float[GetObsSize(1)];
}

WrappedLogAgent::~WrappedLogAgent()
{
    delete[] this->planeBuffer;
}

bboard::Move WrappedLogAgent::act(const bboard::Observation* obs) {
    bboard::Move move = this->agent == nullptr ? bboard::Move::IDLE : this->agent->act(obs);

    // log the (state, action) pair
    if (this->has_buffer()) {
        BoardToPlanes(obs, id, this->planeBuffer);
        this->sampleBuffer->addSample(this->planeBuffer, move);
    }

    return move;
}

void WrappedLogAgent::set_agent(std::unique_ptr<bboard::Agent> agent) {
    if(agent.get() == nullptr) {
        return;
    }

    this->agent = std::move(agent);
}

void WrappedLogAgent::reset() {
    if (this->agent.get() != nullptr) {
        this->agent->id = this->id;
        this->agent->reset();
    }
}

void WrappedLogAgent::release_agent() {
    this->agent.release();
}
