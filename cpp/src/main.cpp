#include <iostream>

#include "runner.h"
#include "ipc_manager.h"
#include "nn/neuralnetapi.h"
#include "nn/tensorrtapi.h"
#include "stateobj.h"

void load_models()
{
    make_unique<TensorrtAPI>(1, 1, "model/", "float32");
    make_unique<TensorrtAPI>(1, 1, "model/", "float16");
    make_unique<TensorrtAPI>(1, 1, "model/", "int8");

    make_unique<TensorrtAPI>(1, 8, "model/", "float32");
    make_unique<TensorrtAPI>(1, 8, "model/", "float16");
    make_unique<TensorrtAPI>(1, 8, "model/", "int8");
}

int main(int argc, char **argv) {
    std::string dataPrefix = "data";
    if (argc >= 2) {
        dataPrefix = argv[1];
    }

    int chunkSize = 1000;
    int chunkCount = 100;

    FileBasedIPCManager ipcManager(dataPrefix, chunkSize, chunkCount);
    Runner runner;

    // generate enough steps (chunkSize * chunkCount) to fill one dataset
    runner.generateSupervisedTrainingData(&ipcManager, 800, -1, chunkSize * chunkCount, false);
    ipcManager.flush();

    load_models();

    return 0;
}
