#ifndef IPCMANAGER_H
#define IPCMANAGER_H

#include "agents/log_agent.h"
#include "episode_info.h"
#include "sample_buffer.h"

// adds compression to z5
#define WITH_BLOSC
#include "z5/types/types.hxx"
#include "z5/filesystem/handle.hxx"
#include "limits.h"

/**
 * @brief The abstract IPCManager class provides an interface to save/transmit (called "write" from now on) training data.
 */
class IPCManager {
public:
    /**
     * @brief writeAgentExperience Write the experience of a single agent episode.
     * @param sampleBuffer The sample buffer which contains the experience of the agent episode.
     * @param agentID The id of the agent which collected this experience.
     * @param maxSteps The maximum amount of steps which will be written from the sample buffer (default: all samples).
     * @returns the number of added samples.
     */
    virtual ulong writeAgentExperience(SampleBuffer& sampleBuffer, const int agentID, const ulong maxSteps=ULONG_MAX) = 0;

    /**
     * @brief writeAgentExperience Write the experience of the given collector.
     * @param collector The SampleCollector which collected experience in an episode.
     * @param maxSteps The maximum amount of steps which will be written from the sample buffer (default: all samples).
     * @returns the number of added samples.
     */
    inline ulong writeAgentExperience(SampleCollector* collector, const ulong maxSteps=ULONG_MAX)
    {
        return writeAgentExperience(*collector->get_buffer(), collector->get_buffer_agent_id(), maxSteps);
    }

    /**
     * @brief writeNewEpisode Writes the episode information of a completed episode. Is expected to be called
     * before writing experience via writeAgentExperience.
     * @param info The info of a completed episode.
     */
    virtual void writeNewEpisode(const EpisodeInfo& info) = 0;

    /**
     * @brief flush When the class is buffering data, flush forces the buffer to save/transmit the buffered data.
     */
    virtual void flush() = 0;
};

/**
 * @brief The FileBasedIPCManager class uses z5 to save logs in files.
 */
class FileBasedIPCManager : public IPCManager {
public:
    /**
     * @brief FileBasedIPCManager Create an IPCManager which stores the logs in files.
     * @param fileNamePrefix The basepath for the logs.
     * @param chunkSize The number of samples which will be saved in one file
     * @param chunkCount The maximum number of chunks of one dataset
     */
    FileBasedIPCManager(std::string fileNamePrefix, int chunkSize, int chunkCount);

    ulong writeAgentExperience(SampleBuffer& sampleBuffer, const int agentID, const ulong maxSteps) override;
    void writeNewEpisode(const EpisodeInfo& info) override;
    void flush() override;

private:
    /**
     * @brief Private struct to store meta-information for an agent episode.
     */
    struct AgentEpisodeInfo {
        int id;
        int steps;
        int episodeID;
    };

    std::string fileNamePrefix;
    unsigned long chunkSize, chunkCount, maxStepCount;

    SampleBuffer sampleBuffer;

    /**
     * @brief datasetStepCount The number of steps which were inserted into the datatset.
     */
    unsigned long datasetStepCount;

    /**
     * @brief processedSteps The total number of processed steps for this dataset.
     */
    unsigned long processedSteps;

    /**
     * @brief fileCount The id of the next file. Used to accelerate getNewFilename.
     */
    unsigned long nextFileId;

    /**
     * @brief activeFile The currently active file (you have to create a new one when the current dataset is full).
     */
    z5::filesystem::handle::File activeFile;

    /**
     * @brief agentEpisodeInfos Stores meta-information for each collected agent experience.
     */
    std::vector<AgentEpisodeInfo> agentEpisodeInfos;

    /**
     * @brief episodeInfos Stores meta-information for the episodes.
     */
    std::vector<EpisodeInfo> episodeInfos;

    /**
     * @brief getNewFileName Get a unique filename for storing dataset containers.
     * @return A filename which starts with fileNamePrefix
     */
    std::string getNewFilename();

    /**
     * @brief createDatasets Creates a zarr group container and initializes the datasets.
     * @param file The file which defines where the zarr group container should be created.
     */
    void createDatasets(z5::filesystem::handle::File file);

    /**
     * @brief flushBuffer Write the content of the sample buffer to the datasets.
     * @param file The file which points to the datasets.
     */
    void flushSampleBuffer(z5::filesystem::handle::File file);
};

#endif // IPCMANAGER_H
