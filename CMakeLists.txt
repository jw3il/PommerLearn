cmake_minimum_required(VERSION 2.8)

set (CMAKE_CXX_STANDARD 17)
project(PommerLearn CXX)

if(DEFINED ENV{CONDA_ENV_PATH})
    MESSAGE(STATUS "CONDA_ENV_PATH set to: $ENV{CONDA_ENV_PATH}")
else()
    MESSAGE(SEND_ERROR "CONDA_ENV_PATH not set")
endif()

# For now, just directly include the pomcpp code
file(GLOB source_files
    "deps/pomcpp/src/bboard/*.cpp"
    "deps/pomcpp/src/agents/*.cpp"
    "cpp/src/*.cpp"
    "cpp/include/*.h"
)

include_directories("cpp/include/")
include_directories("deps/pomcpp/include")
include_directories("$ENV{CONDA_ENV_PATH}/include")

find_package (Threads)

add_executable(${PROJECT_NAME} ${source_files})

find_library(BLOSC_LIB blosc "$ENV{CONDA_ENV_PATH}/lib")
target_link_libraries(${PROJECT_NAME} "${BLOSC_LIB}")

target_link_libraries(${PROJECT_NAME} ${CMAKE_THREAD_LIBS_INIT})
