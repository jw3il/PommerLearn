#ifndef EPISODE_INFO_H
#define EPISODE_INFO_H

/**
 * @brief Contains metadata for an episode.
 */
struct EpisodeInfo {
    int winner;
    bool isDraw;
    bool isDone;
    int steps;
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
