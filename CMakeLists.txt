cmake_minimum_required(VERSION 2.8)

set (CMAKE_CXX_STANDARD 14)
project(PommerLearn CXX)

file(GLOB source_files
    "deps/pomcpp/include/*.hpp"
    "deps/pomcpp/src/*.h"
    "deps/pomcpp/src/*.cpp"
    "deps/pomcpp/src/bboard/*.h"
    "deps/pomcpp/src/bboard/*.cpp"
    "deps/pomcpp/src/agents/*.h"
    "deps/pomcpp/src/agents/*.cpp"
    "*.h"
    "*.cpp"
)

include_directories("cpp")
include_directories("deps/pomcpp/src")
include_directories("deps/pomcpp/include")
include_directories("deps/pomcpp/src/bboard")
include_directories("deps/pomcpp/src/agents")

find_package (Threads)

add_executable(${PROJECT_NAME} ${source_files})

target_link_libraries( ${PROJECT_NAME} ${CMAKE_THREAD_LIBS_INIT} )
