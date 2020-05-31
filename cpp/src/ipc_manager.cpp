#include "ipc_manager.h"
#include "data_representation.h"

#include "nlohmann/json.hpp"
#include "xtensor/xarray.hpp"

// factory functions to create files, groups and datasets
#include "z5/factory.hxx"
// handles for z5 filesystem objects
#include "z5/filesystem/handle.hxx"
// io for xtensor multi-arrays
#include "z5/multiarray/xtensor_access.hxx"
// attribute functionality
#include "z5/attributes.hxx"

FileBasedIPCManager::FileBasedIPCManager(std::string fileName) {
    // TODO initialize buffers
    this->episode = 0;
    this->fileName = fileName;

    z5::filesystem::handle::File f(fileName);

    // for debugging: only create one dataset
    if(f.exists())
        f.remove();
}

void FileBasedIPCManager::writeEpisode(LogAgent* logAgent) {
    // TODO add episode to buffer instead of directly writing to file

    z5::filesystem::handle::File f(this->fileName);

    if(!f.exists())
        z5::createFile(f, true);

    unsigned long episodeSteps = logAgent->step;

    std::string prefix("data_ep");

    // observations

    std::vector<size_t> obs_shape = { episodeSteps, PLANE_COUNT, PLANE_WIDTH, PLANE_HEIGHT};
    std::vector<size_t> obs_chunks = { 100, PLANE_COUNT, PLANE_WIDTH, PLANE_HEIGHT};

    // TODO: Add compression
    // TODO: Use 16 bit floats
    std::string ds_obs_name = prefix.append(std::to_string(this->episode)).append("_obs");
    auto ds_obs = z5::createDataset(f, ds_obs_name, "float32", obs_shape, obs_chunks);

    // TODO: Insert actual episode data
    z5::types::ShapeType obs_offset = { 0, 0, 0, 0, 0 };
    xt::xarray<float> obs_array(obs_shape, 42.0);
    z5::multiarray::writeSubarray<float>(ds_obs, obs_array, obs_offset.begin());

    // actions

    std::vector<size_t> act_shape = { episodeSteps };
    std::vector<size_t> act_chunks = { 100 };

    // TODO: Add compression
    std::string ds_act_name = prefix.append(std::to_string(this->episode)).append("_act");
    auto ds_act = z5::createDataset(f, ds_act_name, "int8", act_shape, act_chunks);

    // TODO: Insert actual episode data
    z5::types::ShapeType act_offset = { 0 };
    xt::xarray<int8_t> act_array(act_shape, 7);
    z5::multiarray::writeSubarray<int8_t>(ds_act, act_array, act_offset.begin());

    this->episode++;
}

void FileBasedIPCManager::flush() {
    // TODO write file buffer
}
