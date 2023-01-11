#ifndef LOGAGENT_H
#define LOGAGENT_H

#include "bboard.hpp"
#include "sample_buffer.h"

/**
 * @brief Interface to allow access to an agent's sample buffer.
 */
class SampleCollector {
public:
    /**
     * @brief create_buffer Create a sample buffer.
     * @param maxEpisodeLength The maximum episode length will determine the capacity of the buffer.
     */
    void create_buffer(int maxEpisodeLength);

    /**
     * @brief get_sample_buffer Get the sample buffer used by this agent to collect experience.
     * @return The agent's sample buffer.
     */
    SampleBuffer* get_buffer();

    /**
     * @brief has_buffer Check whether a sample buffer has been created.
     * @return true iff a buffer has been created
     */
    bool has_buffer() const;

    /**
     * @brief get_buffer_agent_id Get the id of the agent associated with this buffer.
     * @return the agent's id.
     */
    virtual int get_buffer_agent_id() = 0;

    /**
     * @brief Enable or disable the sample collector (default is enabled).
     * 
     * @param enabled whether the sample collector should be enabled
     */
    void set_logging_enabled(bool enabled);

    /**
     * @brief Get whether this sample collector is enabled.
     */
    bool get_logging_enabled();

protected:
    std::unique_ptr<SampleBuffer> sampleBuffer;
    bool enabled = true;
};

/**
 * @brief An agent which can log observations and actions.
 */
class LogAgent : public bboard::Agent, public SampleCollector {
public:
    // SampleCollector
    int get_buffer_agent_id() override;
};

/**
 * @brief Automatically logs the observations and actions of an underlying agent (which does not have to be a log agent).
 */
class WrappedLogAgent : public LogAgent {
public:
    /**
     * @brief LogAgent Create a LogAgent for logging episodes.
     */
    WrappedLogAgent();
    ~WrappedLogAgent();

    /**
     * @brief set_agent Set the controlling agent.
     * @param agent The agent which decides which actions this agent will use.
     */
    void set_agent(std::unique_ptr<bboard::Agent> agent);

    /**
     * @brief delete_agent Delete the agent if it exists.
     */
    void release_agent();

    // bboard::Agent

    bboard::Move act(const bboard::Observation* obs) override;
    void reset() override;

private:
    std::unique_ptr<bboard::Agent> agent;
    float* planeBuffer;
};

#endif // LOGAGENT_H
