cmake_minimum_required(VERSION 2.8)

set (CMAKE_CXX_STANDARD 17)
project(PommerLearn CXX)

option(USE_TENSORRT              "Build with TensorRT support"  ON)

add_definitions(-DMODE_POMMERMAN)

if(DEFINED ENV{CONDA_ENV_PATH})
    MESSAGE(STATUS "CONDA_ENV_PATH set to: $ENV{CONDA_ENV_PATH}")
else()
    MESSAGE(SEND_ERROR "CONDA_ENV_PATH not set")
endif()

if(DEFINED ENV{BLAZE_PATH})
    MESSAGE(STATUS "BLAZE_PATH set to: $ENV{BLAZE_PATH}")
    include_directories("$ENV{BLAZE_PATH}/include")
    include_directories("$ENV{BLAZE_PATH}")
else()
    MESSAGE(SEND_ERROR "BLAZE_PATH not set")
endif()

# For now, just directly include the pomcpp code
file(GLOB source_files
    "deps/pomcpp/src/bboard/*.cpp"
    "deps/pomcpp/src/agents/*.cpp"
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
    "cpp/src/*.cpp"
    "cpp/include/*.h"
)

include_directories("cpp/include/")
include_directories("deps/pomcpp/include")
include_directories("$ENV{CONDA_ENV_PATH}/include")
include_directories("deps/CrazyAra/engine/src")
include_directories("deps/CrazyAra/engine/src/nn/")
include_directories("deps/CrazyAra/engine/src/agents/")

find_package(Threads REQUIRED)

if (USE_TENSORRT)
    # build CrazyAra with TensorRT support, requires a working TensorRT-MXNet library package
    message(STATUS "Enabled TensorRT support")
    message(STATUS "TensorRT path: $ENV{TENSORRT_PATH}")
    # add additional source file
    set(source_files
        ${source_files}
        $ENV{TENSORRT_PATH}/samples/common/logger.cpp
        )
#    find_package(CUDA REQUIRED)
    message(STATUS "CUDA path: $ENV{CUDA_PATH}")
    include_directories("$ENV{CUDA_PATH}/include")
    link_directories("$ENV{CUDA_PATH}/lib64")
    link_directories("$ENV{CUDA_PATH}/lib/x64")
    link_directories("$ENV{TENSORRT_PATH}/lib")
if(WIN32)
  find_library(TENSORRT_LIBRARY_MYELIN myelin64_1
    HINTS  ${TENSORRT_PATH}
    PATH_SUFFIXES lib lib64 lib/x64)
endif()
    include_directories("$ENV{TENSORRT_PATH}/include")
    include_directories("$ENV{TENSORRT_PATH}/samples/common/")
    add_definitions(-DTENSORRT)
endif()

add_executable(${PROJECT_NAME} ${source_files})
# include filesystem (needed for z5)
target_link_libraries(${PROJECT_NAME} stdc++fs)

find_library(BLOSC_LIB blosc "$ENV{CONDA_ENV_PATH}/lib")
target_link_libraries(${PROJECT_NAME} "${BLOSC_LIB}")

target_link_libraries(${PROJECT_NAME} ${CMAKE_THREAD_LIBS_INIT})

find_package( blaze REQUIRED)
if( blaze_FOUND )
     add_library( blaze_target INTERFACE )
     target_link_libraries( blaze_target INTERFACE blaze::blaze )
endif()

if (USE_TENSORRT)
    target_link_libraries(${PROJECT_NAME} nvonnxparser nvinfer cudart myelin ${CUDART_LIB} ${CUBLAS_LIB} ${CUDNN_LIB})
endif()
