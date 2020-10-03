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

#endif // EPISODE_INFO_H
