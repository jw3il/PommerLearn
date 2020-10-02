cmake_minimum_required(VERSION 2.8)

set (CMAKE_CXX_STANDARD 17)
project(PommerLearn CXX)

option(USE_TENSORRT              "Build with TensorRT support"  OFF)
option(USE_TORCH                 "Build with Torch support"  ON)

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
include_directories("$ENV{CONDA_ENV_PATH}/include")

########################
#      Libraries       #
########################

# CrazyAra

add_definitions(-DMODE_POMMERMAN)
# add_definitions(-DDISABLE_UCI_INFO)  # optional

check_env_defined("BLAZE_PATH")
include_directories("$ENV{BLAZE_PATH}/include")
include_directories("$ENV{BLAZE_PATH}")

if (USE_TENSORRT)
    # build CrazyAra with TensorRT support
    message(STATUS "Enabled TensorRT Backend")
    check_env_defined("TENSORRT_PATH")
    check_env_defined("CUDA_PATH")
    include_directories("$ENV{CUDA_PATH}/include")
    include_directories("$ENV{TENSORRT_PATH}/include")
    include_directories("$ENV{TENSORRT_PATH}/samples/common/")
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

add_library(CrazyAra ${CrazyAra_source_files})

if (USE_TORCH)
    target_link_libraries(CrazyAra "${TORCH_LIBRARIES}")
endif()

find_package(Threads REQUIRED)

if (USE_TENSORRT)
    target_link_libraries(CrazyAra -L"$ENV{CUDA_PATH}/lib64")
    target_link_libraries(CrazyAra -L"$ENV{CUDA_PATH}/lib/x64")
    target_link_libraries(CrazyAra -L"$ENV{TENSORRT_PATH}/lib")

    target_link_libraries(CrazyAra nvonnxparser nvinfer cudart myelin)
endif()

# pomcpp

# For now, just directly include the pomcpp code
file(GLOB pomcpp_source_files
    "deps/pomcpp/src/bboard/*.cpp"
    "deps/pomcpp/src/agents/*.cpp"
)

include_directories("deps/pomcpp/include")
add_library(pomcpp ${pomcpp_source_files})

########################
#    Main Target(s)    #
########################

# blosc needed for z5
find_library(BLOSC blosc "$ENV{CONDA_ENV_PATH}/lib")

file(GLOB source_files
    "cpp/src/*.cpp"
    "cpp/include/*.h"
)

include_directories("cpp/include/")

add_executable(${PROJECT_NAME} ${source_files})

if(APPLE)
    # apple has trouble linking to stdc++fs somehow
    target_link_libraries(${PROJECT_NAME} ${CMAKE_THREAD_LIBS_INIT} ${BLOSC} CrazyAra pomcpp)
else()
    target_link_libraries(${PROJECT_NAME} ${CMAKE_THREAD_LIBS_INIT} stdc++fs ${BLOSC} CrazyAra pomcpp)
endif()


target_link_libraries(${PROJECT_NAME} ${CMAKE_THREAD_LIBS_INIT} ${BLOSC} CrazyAra pomcpp)

# Testing

set(test_name ${PROJECT_NAME}Test)

set(source_files_nomain ${source_files})
list(FILTER source_files_nomain EXCLUDE REGEX "main.cpp$")

file(GLOB test_files
    "cpp/test/*.cpp"
)

add_executable(${test_name} ${source_files_nomain} ${test_files})

if(APPLE)
    target_link_libraries(${test_name} ${CMAKE_THREAD_LIBS_INIT} ${BLOSC} CrazyAra pomcpp)
else()
    target_link_libraries(${test_name} ${CMAKE_THREAD_LIBS_INIT} stdc++fs ${BLOSC} CrazyAra pomcpp)
endif()
