#ifndef IPCMANAGER_H
#define IPCMANAGER_H

#include "log_agent.h"

// adds compression to z5
#define WITH_BLOSC
#include "z5/types/types.hxx"

/**
 * @brief The abstract IPCManager class provides an interface to save/transmit (called "write" from now on) training data.
 */
class IPCManager
{
public:
    /**
     * @brief writeEpisode Write the episode of the given LogAgent.
     * @param logAgent The LogAgent which contains an episode log.
     */
    virtual void writeEpisode(LogAgent* logAgent) = 0;

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
     * @param fileName The basepath/archive name for the logs.
     */
    FileBasedIPCManager(std::string fileName);
    void writeEpisode(LogAgent* logAgent);
    void flush();
private:
    int episode;
    std::string fileName;
    std::string compressor;
    z5::types::CompressionOptions compressionOptions;
};

#endif // IPCMANAGER_H
