#ifndef EPISODE_INFO_H
#define EPISODE_INFO_H

#include "bboard.hpp"

/**
 * @brief Contains metadata for an episode.
 */
struct EpisodeInfo {
    bboard::State initialState;
    int winningAgent;
    int winningTeam;
    bool isDraw;
    bool isDone;
    int steps;
    bool dead[bboard::AGENT_COUNT];
};

/**
 * @brief Contains metadata for an episode as seen by a single agent.
 */
struct AgentEpisodeInfo {
   int agentId;
   int steps;
   int episode;
};

#endif // EPISODE_INFO_H
