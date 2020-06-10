#include <iostream>

#include "runner.h"
#include "ipc_manager.h"

int main() {
    int chunkSize = 1000;
    int chunkCount = 100;

    FileBasedIPCManager ipcManager("data", chunkSize, chunkCount);
    Runner runner;

    // generate enough steps (chunkSize * chunkCount) to fill one dataset
    runner.generateSupervisedTrainingData(&ipcManager, 500, -1, chunkSize * chunkCount);
    ipcManager.flush();

    return 0;
}
