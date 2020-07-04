#include <iostream>

#include "runner.h"
#include "ipc_manager.h"

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
    runner.generateSupervisedTrainingData(&ipcManager, 800, -1, chunkSize * chunkCount);
    ipcManager.flush();

    return 0;
}
