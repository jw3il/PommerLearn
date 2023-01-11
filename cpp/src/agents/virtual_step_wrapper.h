#ifndef VIRTUAL_STEP_WRAPPER_H
#define VIRTUAL_STEP_WRAPPER_H

#include "bboard.hpp"

/**
 * @brief Wraps VirtualStep around any agent type. The agent will receive observations that have been augmented with VirtualStep.
 * 
 * @tparam AgentType the type of the agent
 * @tparam Args argument types of the constructor of that agent
 */
template<class AgentType, class... Args> 
class VirtualStepWrapper : public AgentType {
private:
    // internal state
    bboard::State state;
    // temporary observation
    bboard::Observation tmpObs;
    bboard::ObservationParameters params;

public:
    /**
     * @brief Construct a new object of type VirtualStepWrapper : public AgentType.
     * 
     * @param args the arguments for the constructor of AgentType.
     */
    VirtualStepWrapper(Args&&... args) : AgentType(std::forward<Args>(args)...){
        static_assert(std::is_base_of<bboard::Agent, AgentType>::value, "AgentType not derived from bboard::Agent");
        
        // set parameters to keep all information from the given observations
        params.agentInfoVisibility = bboard::AgentInfoVisibility::All;
        params.agentPartialMapView = false;
        params.exposePowerUps = true;
    }

    // bboard::Agent
    bboard::Move act(const bboard::Observation* obs) override {
        if (obs->timeStep == 0) {
            // reset internal state
            this->state = bboard::State();
            obs->ToState(this->state);
        }
        else {
            // run virtual step on internal state
            obs->VirtualStep(this->state, true, true, nullptr);
        }
        
        // convert state to observation and act
        bboard::Observation::Get(this->state, this->id, this->params, this->tmpObs);
        return AgentType::act(&this->tmpObs);
    }
};

#endif // VIRTUAL_STEP_WRAPPER_H