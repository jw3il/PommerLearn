#include <iostream>
#include <array>

#include "runner.h"
#include "ipc_manager.h"
#include "nn/neuralnetapi.h"
#include "nn/tensorrtapi.h"
#include "agents/rawnetagent.h"
#include "pommerman_raw_net_agent.h"
#include "stateobj.h"

#include "agents.hpp"

void load_models()
{
    make_unique<TensorrtAPI>(1, 1, "model/", "float32");
    make_unique<TensorrtAPI>(1, 1, "model/", "float16");
    make_unique<TensorrtAPI>(1, 1, "model/", "int8");

    make_unique<TensorrtAPI>(1, 8, "model/", "float32");
    make_unique<TensorrtAPI>(1, 8, "model/", "float16");
    make_unique<TensorrtAPI>(1, 8, "model/", "int8");
}

void free_for_all_tourney(size_t nbGames)
{
    Constants::init(false);
    TensorrtAPI nn(1,1, "model", "float32");
    PlaySettings playSettings;
    RawNetAgent rawNetAgent(&nn, &playSettings, true);

    bboard::Environment env;
    PommermanState pommermanState;
    PommermanRawNetAgent pommerRawAgent(&rawNetAgent, &pommermanState);

    srand(time(0));
    std::array<bboard::Agent*, bboard::AGENT_COUNT> agents = {&pommerRawAgent,
                                                              new agents::SimpleAgent(rand()),
                                                              new agents::SimpleAgent(rand()),
                                                              new agents::SimpleAgent(rand()),
                                                             };
    std::array<size_t, 4> nbWins = {0,0,0,0};
    size_t nbDraws = 0;

    for (size_t curIt = 0; curIt < nbGames; ++curIt) {
        env.MakeGame(agents, bboard::GameMode::FreeForAll, rand(), true);

        Runner runner;
        EpisodeInfo episodeInfo = runner.run(env, 800, false);
        if (episodeInfo.winningAgent != -1) {
            nbWins[episodeInfo.winningAgent] += 1;
        }
        else {
            ++nbDraws;
        }
    }

    for (size_t agentIdx = 0; agentIdx < bboard::AGENT_COUNT; ++agentIdx) {
        std::cout << "Agent " << agentIdx << ": " << nbWins[agentIdx] << " wins" << std::endl;
    }

    std::cout << "Draws: " << nbDraws << std::endl;
}

void generate_sl_data(const std::string& dataPrefix)
{
    int chunkSize = 1000;
    int chunkCount = 100;

    FileBasedIPCManager ipcManager(dataPrefix, chunkSize, chunkCount);
    Runner runner;

    // generate enough steps (chunkSize * chunkCount) to fill one dataset
    runner.generateSupervisedTrainingData(&ipcManager, 800, -1, chunkSize * chunkCount, false);
    ipcManager.flush();

    load_models();
}

int main(int argc, char **argv) {
    std::string dataPrefix = "data";
    if (argc >= 2) {
        dataPrefix = argv[1];
    }

//    generate_sl_data(dataPrefix);
    free_for_all_tourney(100);

    return 0;
}
