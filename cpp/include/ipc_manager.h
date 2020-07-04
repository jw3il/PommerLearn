#ifndef IPCMANAGER_H
#define IPCMANAGER_H

#include "log_agent.h"
#include "episode_info.h"
#include "sample_buffer.h"

// adds compression to z5
#define WITH_BLOSC
#include "z5/types/types.hxx"
#include "z5/filesystem/handle.hxx"

/**
 * @brief The abstract IPCManager class provides an interface to save/transmit (called "write" from now on) training data.
 */
class IPCManager {
public:
    /**
     * @brief writeEpisode Write the collected experience of the given LogAgent.
     * @param logAgent The LogAgent which contains an experience log.
     * @param info The info of the corresponding episode.
     */
    virtual void writeAgentExperience(LogAgent* logAgent, EpisodeInfo info) = 0;

    /**
     * @brief writeEpisodeInfo Write the episode information of a completed episode. Is expected to be called before writeAgentExperience.
     * @param info The info of the complete episode.
     */
    virtual void writeEpisodeInfo(EpisodeInfo info) = 0;

    /**
     * @brief flush When the class is buffering logs, flush forces the buffer to save/transmit the buffered data.
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

    void writeAgentExperience(LogAgent* logAgent, EpisodeInfo info);
    void writeEpisodeInfo(EpisodeInfo info);
    void flush();

private:
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
