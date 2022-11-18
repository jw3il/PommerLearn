#ifndef EPISODE_INFO_H
#define EPISODE_INFO_H

#include "bboard.hpp"

/**
 * @brief Contains metadata for an episode.
 */
struct EpisodeInfo
{
    // information about the episode itself
    bboard::State initialState;
    std::array<std::vector<int8_t>, bboard::AGENT_COUNT> actions;
    bboard::GameMode gameMode;

    // information about the result
    int winningAgent;
    int winningTeam;
    bool isDraw;
    bool isDone;
    int steps;
    std::array<bool, bboard::AGENT_COUNT> dead;
};

#endif // EPISODE_INFO_H
