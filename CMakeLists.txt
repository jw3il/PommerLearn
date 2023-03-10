cmake_minimum_required(VERSION 2.8.12)

set (CMAKE_CXX_STANDARD 17)
project(PommerLearn CXX)

option(USE_TENSORRT              "Build with TensorRT support"  ON)
option(USE_TENSORRT8             "Build with TensorRT support (version 8)"  OFF)
option(USE_TORCH                 "Build with Torch support"  OFF)
option(USE_PROFILING             "Build with profiling"   OFF)
option(USE_1VS1_MCTS             "Build with adaptive 1vs1 mcts variant"   OFF)

if(USE_TENSORRT8)
    set(USE_TENSORRT ON)
endif()

function(check_env_defined name)
    if(DEFINED ENV{${name}})
        MESSAGE(STATUS "Environment variable ${name} set to: $ENV{${name}}")
    else()
        MESSAGE(SEND_ERROR "Environment variable ${name} not defined!")
    endif()
endfunction()

########################
# General Dependencies #
########################

# conda-forge headers

check_env_defined("CONDA_ENV_PATH")
include_directories(SYSTEM "$ENV{CONDA_ENV_PATH}/include")

# xtensor and gcc > 5.1 https://xtensor.readthedocs.io/en/latest/compilers.html
# add_definitions(-DXTENSOR_GLIBCXX_USE_CXX11_ABI)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)

########################
#      Libraries       #
########################

# CrazyAra

add_definitions(-DMODE_POMMERMAN)
# add_definitions(-DCRAZYARA_AGENT_PV)
add_definitions(-DDISABLE_UCI_INFO)  # optional
add_definitions(-DMCTS_STORE_STATES)
if(NOT USE_1VS1_MCTS)
    # add single player definition
    add_definitions(-DMCTS_SINGLE_PLAYER)
endif()
add_definitions(-DYNAMIC_NN_ARCH)

check_env_defined("BLAZE_PATH")
include_directories(SYSTEM "$ENV{BLAZE_PATH}")

if (USE_TENSORRT)
    # build CrazyAra with TensorRT support
    message(STATUS "Enabled TensorRT Backend")
    check_env_defined("TENSORRT_PATH")
    check_env_defined("CUDA_PATH")
    include_directories("$ENV{CUDA_PATH}/include")
    include_directories("$ENV{TENSORRT_PATH}/include")
    include_directories("$ENV{TENSORRT_PATH}/samples/common/")
    include_directories("$ENV{TENSORRT_PATH}/parsers/onnx")
    add_definitions(-DTENSORRT)
endif()

if (USE_TORCH)
    message(STATUS "Enabled Torch Backend")
    find_package(Torch REQUIRED)
    add_definitions(-DTORCH)
endif()

file(GLOB CrazyAra_source_files
    "deps/CrazyAra/engine/src/*.cpp"
    "deps/CrazyAra/engine/src/*.h"
    "deps/CrazyAra/engine/src/util/*.h"
    "deps/CrazyAra/engine/src/util/*.cpp"
    "deps/CrazyAra/engine/src/domain/crazyhouse/*.h"
    "deps/CrazyAra/engine/src/domain/crazyhouse/*.cpp"
    "deps/CrazyAra/engine/src/agents/*.cpp"
    "deps/CrazyAra/engine/src/agents/*.h"
    "deps/CrazyAra/engine/src/agents/config/*.cpp"
    "deps/CrazyAra/engine/src/agents/config/*.h"
    "deps/CrazyAra/engine/src/agents/util/*.cpp"
    "deps/CrazyAra/engine/src/agents/util/*.h"
    "deps/CrazyAra/engine/src/nn/*.cpp"
    "deps/CrazyAra/engine/src/nn/*.h"
    "deps/CrazyAra/engine/src/nn/util/*.cpp"
    "deps/CrazyAra/engine/src/nn/util/*.h"
    "deps/CrazyAra/engine/src/manager/*.cpp"
    "deps/CrazyAra/engine/src/manager/*.h"
    "deps/CrazyAra/engine/src/rl/*.cpp"
    "deps/CrazyAra/engine/src/rl/*.h"
)

# ignore most library warnings
set_source_files_properties(
  "$ENV{CONDA_ENV_PATH}/include" "$ENV{BLAZE_PATH}"
  PROPERTIES
  COMPILE_FLAGS "-w"
)

if (USE_TENSORRT)
    # add additional source file
    set(CrazyAra_source_files
        ${CrazyAra_source_files}
        $ENV{TENSORRT_PATH}/samples/common/logger.cpp
    )
endif()

include_directories("deps/CrazyAra/engine/src")
include_directories("deps/CrazyAra/engine/src/nn/")
include_directories("deps/CrazyAra/engine/src/agents/")

add_library(CrazyAra SHARED ${CrazyAra_source_files})
# ignore warnings in CrazyAra
target_compile_options(CrazyAra PRIVATE "-w")

if (USE_TORCH)
    target_link_libraries(CrazyAra "${TORCH_LIBRARIES}")
endif()

find_package(Threads REQUIRED)

if (USE_TENSORRT)
    target_link_libraries(CrazyAra -L"$ENV{CUDA_PATH}/lib64")
    target_link_libraries(CrazyAra -L"$ENV{CUDA_PATH}/lib/x64")
    target_link_libraries(CrazyAra -L"$ENV{TENSORRT_PATH}/lib")

    if (USE_TENSORRT8)
        target_link_libraries(CrazyAra nvonnxparser nvinfer cudart)
    else()
        target_link_libraries(CrazyAra nvonnxparser nvinfer cudart myelin)
    endif()
endif()

# -pg performance profiling flags
if (USE_PROFILING)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
    SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")
endif()

# pomcpp

# For now, just directly include the pomcpp code
file(GLOB pomcpp_source_files
    "deps/pomcpp/src/bboard/*.cpp"
    "deps/pomcpp/src/agents/*.cpp"
)

include_directories("deps/pomcpp/include")
add_library(pomcpp SHARED ${pomcpp_source_files})

########################
#    Main Target(s)    #
########################

# blosc needed for z5
find_library(BLOSC blosc "$ENV{CONDA_ENV_PATH}/lib")

# boost
set(Boost_Dir "$ENV{CONDA_ENV_PATH}/lib")
set(Boost_INCLUDE_DIR "$ENV{CONDA_ENV_PATH}/include")
FIND_PACKAGE(Boost COMPONENTS program_options REQUIRED)

file(GLOB source_files
    "cpp/src/*.cpp"
    "cpp/src/*.h"
    "cpp/src/agents/*.cpp"
    "cpp/src/agents/*.h"
)

set(source_files_nomain ${source_files})
list(FILTER source_files_nomain EXCLUDE REGEX "main.cpp$")

include_directories("cpp/src/")

# Main executable

add_executable(${PROJECT_NAME} ${source_files})
target_link_libraries(${PROJECT_NAME} ${CMAKE_THREAD_LIBS_INIT} ${BLOSC} CrazyAra pomcpp Boost::program_options)
# we need need to link filesystem library manually.
target_link_libraries(CrazyAra stdc++fs)

# Library with Python interface

set(lib_name ${PROJECT_NAME}Py)

file(GLOB lib_files
    "deps/pomcpp/src/pyinterface/from_json.cpp"
    "deps/pomcpp/src/pyinterface/pymethods.cpp"
    "cpp/src/pyinterface/new_agent.cpp"
)
add_library(${lib_name} SHARED ${source_files_nomain} ${lib_files})
target_link_libraries(${lib_name} ${CMAKE_THREAD_LIBS_INIT} ${BLOSC} CrazyAra pomcpp)
