#include "ipc_manager.h"
#include "data_representation.h"

#include <algorithm>

#include "xtensor/xarray.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xadapt.hpp"

#include "nlohmann/json.hpp"

#include "z5/dataset.hxx"
#include "z5/factory.hxx"
#include "z5/multiarray/xtensor_access.hxx"
#include "z5/attributes.hxx"

FileBasedIPCManager::FileBasedIPCManager(std::string fileNamePrefix, int chunkSize, int chunkCount)
    : fileNamePrefix(fileNamePrefix), chunkSize(chunkSize), chunkCount(chunkCount),
      sampleBuffer(chunkSize), nextFileId(0), activeFile(getNewFilename())
{
    this->maxStepCount = chunkSize * chunkCount;
    this->processedSteps = 0;
    this->datasetStepCount = 0;
}

void FileBasedIPCManager::flushSampleBuffer(z5::filesystem::handle::File file) {
    SampleBuffer& buffer = this->sampleBuffer;
    ulong count = buffer.getCount();

    // only write when we actually got new data
    if (count <= 0)
        return;

    if (!file.exists()) {
        // create a new dataset when we just started logging and there is no file yet
        createDatasets(file);
    }

    // observations

    std::vector<size_t> obsShape = { count, PLANE_COUNT, PLANE_SIZE, PLANE_SIZE };
    z5::types::ShapeType obsOffset = { this->datasetStepCount, 0, 0, 0, 0 };

    auto xtObs = xt::adapt(buffer.getObs(), buffer.getTotalObsValCount(), xt::no_ownership(), obsShape);
    auto obsDataset = z5::openDataset(file, "obs");
    z5::multiarray::writeSubarray<float>(obsDataset, xtObs, obsOffset.begin());

    // actions

    std::vector<size_t> actShape = { count };
    z5::types::ShapeType actOffset = { this->datasetStepCount };

    auto xtAct = xt::adapt(buffer.getAct(), count, xt::no_ownership(), actShape);
    auto actDataset = z5::openDataset(file, "act");
    z5::multiarray::writeSubarray<int8_t>(actDataset, xtAct, actOffset.begin());

    // policy

    std::vector<size_t> polShape = { count, NUM_MOVES };
    z5::types::ShapeType polOffset = { this->datasetStepCount, 0 };

    auto xtPol = xt::adapt(buffer.getPol(), count * NUM_MOVES, xt::no_ownership(), polShape);
    auto polDataset = z5::openDataset(file, "pol");
    z5::multiarray::writeSubarray<float>(polDataset, xtPol, polOffset.begin());

    // values

    std::vector<size_t> valShape = { count };
    z5::types::ShapeType valOffset = { this->datasetStepCount };

    auto xtVal = xt::adapt(buffer.getVal(), count, xt::no_ownership(), valShape);
    auto valDataset = z5::openDataset(file, "val");
    z5::multiarray::writeSubarray<float>(valDataset, xtVal, valOffset.begin());

    // remember that we added the samples and clear the buffer
    this->datasetStepCount += buffer.getCount();
    buffer.clear();
}

void FileBasedIPCManager::writeAgentExperience(SampleBuffer& sampleBuffer, const int agentID)
{
    if (sampleBuffer.getCount() == 0)
        return;

    if (this->processedSteps == this->maxStepCount) {
        // when the current dataset is full and we want to add more data..

        // first flush the current content/metainformation
        flush();

        // obtain the name of the next file
        this->activeFile = z5::filesystem::handle::File(getNewFilename());

        // clear meta information
        this->agentEpisodeInfos.clear();
        this->episodeInfos.clear();

        this->processedSteps = 0;
        this->datasetStepCount = 0;
    }

    // TODO: Adapt value for team mode
    EpisodeInfo& lastInfo = this->episodeInfos.back();
    float value = lastInfo.winningAgent == agentID ? 1.0f : (lastInfo.dead[agentID] ? -1.0f : 0.0f);
    sampleBuffer.setValues(value);

    // compute the amount of steps we are allowed to insert into this dataset
    // TODO: Maybe insert remaining steps into new dataset
    ulong trimmedSteps = std::min(sampleBuffer.getCount(), this->maxStepCount - this->processedSteps);
    ulong currentStep = 0;
    ulong remainingSteps = trimmedSteps;

    while (remainingSteps > 0) {
        // add the samples of the agent to the global samplebuffer
        ulong steps = this->sampleBuffer.addSamples(sampleBuffer, currentStep, remainingSteps);

        if (steps < remainingSteps) {
            // buffer is full, flush it
            this->flushSampleBuffer(this->activeFile);
        }

        currentStep += steps;
        remainingSteps -= steps;
    }

    // add meta information of this agent episode
    AgentEpisodeInfo agentInfo;
    agentInfo.id = agentID;
    agentInfo.steps = sampleBuffer.getCount();
    agentInfo.episodeID = this->episodeInfos.size() - 1;

    this->agentEpisodeInfos.push_back(agentInfo);

    this->processedSteps += trimmedSteps;
}

void FileBasedIPCManager::writeNewEpisode(const EpisodeInfo& info)
{
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
    if (!this->activeFile.exists())
        return;

    this->flushSampleBuffer(this->activeFile);

    nlohmann::json attributes;

    // TODO: Create all attribute arrays in one pass

    // agents
    attributes["AgentIds"] = _mapVector<AgentEpisodeInfo, int>(agentEpisodeInfos, [](AgentEpisodeInfo &info){ return info.id;});
    attributes["AgentSteps"] = _mapVector<AgentEpisodeInfo, int>(agentEpisodeInfos, [](AgentEpisodeInfo &info){ return info.steps;});
    attributes["AgentEpisode"] =  _mapVector<AgentEpisodeInfo, int>(agentEpisodeInfos, [](AgentEpisodeInfo &info){ return info.episodeID;});

    // episodes
    attributes["EpisodeInitialState"] = _mapVector<EpisodeInfo, std::string>(episodeInfos, [](EpisodeInfo &info){ return InitialStateToString(info.initialState);});
    attributes["EpisodeWinner"] = _mapVector<EpisodeInfo, int>(episodeInfos, [](EpisodeInfo &info){ return info.winningAgent;});
    attributes["EpisodeSteps"] = _mapVector<EpisodeInfo, int>(episodeInfos, [](EpisodeInfo &info){ return info.steps;});

    // total steps
    attributes["Steps"] = this->processedSteps;

    z5::writeAttributes(this->activeFile, attributes);
}

std::string FileBasedIPCManager::getNewFilename() {
    std::string currentFilename;

    do {
        // if the filename is not empty, this must be at least the second iteration of the loop
        // => there exists a file with name currentFilename
        if (currentFilename.size() != 0) {
            this->nextFileId++;
        }

        // generate new filename
        currentFilename = this->fileNamePrefix + "_" + std::to_string(this->nextFileId) + ".zr";

    } while (std::filesystem::exists(currentFilename));

    return currentFilename;
}

void FileBasedIPCManager::createDatasets(z5::filesystem::handle::File file) {
    if (file.exists()) {
        throw "Datasets at '" + file.path().string() + "' already exist!";
    }

    z5::createFile(file, true);

    // use compression (fixed value for now)
    std::string compressor = "blosc";
    z5::types::CompressionOptions compressionOptions;

    // default compression options
    compressionOptions["codec"] = (std::string)"lz4";
    compressionOptions["level"] = (int)5;
    compressionOptions["shuffle"] = (int)1;

    // create datasets

    // observations

    std::vector<size_t> obsShape = { this->maxStepCount, PLANE_COUNT, PLANE_SIZE, PLANE_SIZE};
    std::vector<size_t> obsChunks = { this->chunkSize, PLANE_COUNT, PLANE_SIZE, PLANE_SIZE};
    z5::createDataset(file, "obs", "float32", obsShape, obsChunks, compressor, compressionOptions);

    // actions

    std::vector<size_t> actShape = { this->maxStepCount };
    std::vector<size_t> actChunks = { this->chunkSize };
    z5::createDataset(file, "act", "int8", actShape, actChunks, compressor, compressionOptions);

    // policy

    std::vector<size_t> polShape = { this->maxStepCount, NUM_MOVES };
    std::vector<size_t> polChunks = { this->chunkSize, NUM_MOVES };
    z5::createDataset(file, "pol", "float32", polShape, polChunks, compressor, compressionOptions);

    // values

    std::vector<size_t> valShape = { this->maxStepCount };
    std::vector<size_t> valChunks = { this->chunkSize };
    z5::createDataset(file, "val", "float32", valShape, valChunks, compressor, compressionOptions);
}
