#include <iostream>

#include "runner.h"
#include "ipc_manager.h"

int main() {
    FileBasedIPCManager ipcManager("data.zr");
    Runner runner;

    runner.generateSupervisedTrainingData(&ipcManager, 500, 100);

    return 0;
}
