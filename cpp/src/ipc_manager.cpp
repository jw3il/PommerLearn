#include "ipc_manager.h"
#include "data_representation.h"

#include "nlohmann/json.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xview.hpp"

#include <algorithm>

#include "z5/dataset.hxx"
// factory functions to create files, groups and datasets
#include "z5/factory.hxx"
// handles for z5 filesystem objects
#include "z5/filesystem/handle.hxx"
// io for xtensor multi-arrays
#include "z5/multiarray/xtensor_access.hxx"
// attribute functionality
#include "z5/attributes.hxx"

FileBasedIPCManager::FileBasedIPCManager(std::string fileNamePrefix, int chunkSize, int chunkCount)
    : fileNamePrefix(fileNamePrefix), chunkSize(chunkSize), chunkCount(chunkCount) {

    this->maxStepCount = chunkSize * chunkCount;
    this->nextFileId = 0;
    this->stepCount = 0;

    this->activeFileName = getNewFileName();
}

/**
 * @brief _writeEpisodeSteps Converts and copies the collected experience of the logAgent at slice [agentStepOffset, agentStepOffset + count] into the datasets at [datasetStepOffset, datasetStepOffset + count].
 * @param file A file which contains the datasets
 * @param logAgent The agent which holds the log information
 * @param datasetStepOffset The step offset in the datasets
 * @param agentStepOffset The step offset in the agent's experience
 * @param count The number of steps
 * @param value The value of the episode (to be replaced by individual values for each step)
 */
void _writeEpisodeSteps(z5::filesystem::handle::File &file, LogAgent* logAgent, uint datasetStepOffset, uint agentStepOffset, uint count, float value) {

    // observations

    std::vector<size_t> obs_shape = { count, PLANE_COUNT, PLANE_SIZE, PLANE_SIZE};
    z5::types::ShapeType obs_offset = { datasetStepOffset, 0, 0, 0, 0 };
    xt::xarray<float> obs_array(obs_shape);

    for (uint i = 0; i < count; i++) {
        // convert observation planes for state at index (agentStepOffset + i)
        // and insert them at position i into obs_array
        bboard::State state = logAgent->stateBuffer[agentStepOffset + i];
        StateToPlanes(state, logAgent->id, obs_array, i);
    }

    // TODO: Maybe keep datasets open?
    auto ds_obs = z5::openDataset(file, "obs");
    z5::multiarray::writeSubarray<float>(ds_obs, obs_array, obs_offset.begin());

    // actions

    std::vector<size_t> act_shape = { count };
    z5::types::ShapeType act_offset = { datasetStepOffset };
    xt::xarray<int8_t> act_array(act_shape);

    for (uint i = 0; i < count; i++) {
        act_array[i] = MoveToInt(logAgent->actionBuffer[agentStepOffset + i]);
    }

    auto ds_act = z5::openDataset(file, "act");
    z5::multiarray::writeSubarray<int8_t>(ds_act, act_array, act_offset.begin());

    // values

    std::vector<size_t> val_shape = { count };
    z5::types::ShapeType val_offset = { datasetStepOffset };
    xt::xarray<float> val_array(val_shape, value);

    auto ds_val = z5::openDataset(file, "val");
    z5::multiarray::writeSubarray<float>(ds_val, val_array, val_offset.begin());
}

void FileBasedIPCManager::writeAgentExperience(LogAgent* logAgent, EpisodeInfo info) {
    if (logAgent->step == 0)
        return;

    if (this->stepCount == this->maxStepCount) {
        // when the current dataset is full and we want to add more data..

        // first flush the current content/metainformation
        flush();

        // obtain the name of the next file
        this->activeFileName = getNewFileName();

        // clear meta information
        this->agentEpisodeInfos.clear();
        this->episodeInfos.clear();

        this->stepCount = 0;
    }

    z5::filesystem::handle::File file(this->activeFileName);

    if (this->stepCount == 0 && !file.exists()) {
        // create a new dataset when we just started logging and there is no file yet
        createDatasets(this->activeFileName);
    }

    // compute the amount of steps we are allowed to insert into this dataset
    // TODO: Maybe insert remaining steps into new dataset
    uint trimmedSteps = std::min(logAgent->step, (uint)(this->maxStepCount - this->stepCount));
    float value = info.winner == logAgent->id ? 1.0f : (info.dead[logAgent->id] ? -1.0f : 0.0f);

    // TODO: Maybe add steps to a buffer first
    _writeEpisodeSteps(file, logAgent, this->stepCount, 0, trimmedSteps, value);

    // add meta information
    AgentEpisodeInfo agentEpisodeInfo;
    agentEpisodeInfo.steps = logAgent->step;
    agentEpisodeInfo.agentId = logAgent->id;
    agentEpisodeInfo.episode = this->episodeInfos.size() - 1;
    this->agentEpisodeInfos.push_back(agentEpisodeInfo);

    this->stepCount += trimmedSteps;
}

void FileBasedIPCManager::writeEpisodeInfo(EpisodeInfo info) {
    this->episodeInfos.push_back(info);
}


template<typename A, typename B>
/**
 * @brief _mapVector Maps a vector<A> element-wise to a vector<B>.
 * @param vectorA The source vector
 * @param mapAToB A function which maps elements from type A to elements of type B
 * @return The new vector<B>
 */
std::vector<B> _mapVector(std::vector<A> vectorA, std::function<B(A&)> mapAToB) {
    std::vector<B> vectorB;

    std::transform(
        vectorA.begin(),
        vectorA.end(),
        std::back_inserter(vectorB),
        mapAToB
    );

    return vectorB;
}


void FileBasedIPCManager::flush() {
    z5::filesystem::handle::File f(this->activeFileName);

    if (!f.exists())
        return;

    nlohmann::json attributes;

    // TODO: Create all attribute arrays in one pass

    // agents
    attributes["AgentIds"] = _mapVector<AgentEpisodeInfo, int>(agentEpisodeInfos, [](AgentEpisodeInfo &info){ return info.agentId;});
    attributes["AgentSteps"] = _mapVector<AgentEpisodeInfo, int>(agentEpisodeInfos, [](AgentEpisodeInfo &info){ return info.steps;});
    attributes["AgentEpisode"] = _mapVector<AgentEpisodeInfo, int>(agentEpisodeInfos, [](AgentEpisodeInfo &info){ return info.episode;});

    // episodes
    attributes["EpisodeWinner"] = _mapVector<EpisodeInfo, int>(episodeInfos, [](EpisodeInfo &info){ return info.winner;});
    attributes["EpisodeSteps"] = _mapVector<EpisodeInfo, int>(episodeInfos, [](EpisodeInfo &info){ return info.steps;});

    // total steps
    attributes["Steps"] = this->stepCount;

    z5::writeAttributes(f, attributes);
}

std::string FileBasedIPCManager::getNewFileName() {
    std::string fileName = this->activeFileName;

    // obtain a new unique filename which does not exist yet (and has a higher fileCount number)
    while (fileName.size() == 0 || std::filesystem::exists(fileName)) {
        fileName = this->fileNamePrefix + "_" + std::to_string(this->nextFileId) + ".zr";
        // remember the current filecount so that the next call will be faster
        this->nextFileId++;
    }

    return fileName;
}

void FileBasedIPCManager::createDatasets(std::string fileName) {
    z5::filesystem::handle::File f(fileName);

    if (f.exists()) {
        throw "Datasets at '" + fileName + "' already exist!";
    }

    z5::createFile(f, true);

    // use compression (fixed value for now)
    std::string compressor = "blosc";
    z5::types::CompressionOptions compressionOptions;

    // default compression options
    compressionOptions["codec"] = (std::string)"lz4";
    compressionOptions["level"] = (int)5;
    compressionOptions["shuffle"] = (int)1;

    // create datasets

    // observations

    std::vector<size_t> obs_shape = { this->maxStepCount, PLANE_COUNT, PLANE_SIZE, PLANE_SIZE};
    std::vector<size_t> obs_chunks = { this->chunkSize, PLANE_COUNT, PLANE_SIZE, PLANE_SIZE};

    std::unique_ptr<z5::Dataset> ds_obs = z5::createDataset(f, "obs", "float32", obs_shape, obs_chunks, compressor, compressionOptions);

    // actions (not chunked)

    std::vector<size_t> act_shape = { this->maxStepCount };
    z5::createDataset(f, "act", "int8", act_shape, act_shape, compressor, compressionOptions);

    // values (not chunked)

    std::vector<size_t> val_shape = { this->maxStepCount };
    z5::createDataset(f, "val", "float32", val_shape, val_shape, compressor, compressionOptions);
}
