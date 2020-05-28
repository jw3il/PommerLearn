cmake_minimum_required(VERSION 2.8)

set (CMAKE_CXX_STANDARD 14)
project(PommerLearn CXX)

# For now, just directly include the pomcpp code
file(GLOB source_files
    "deps/pomcpp/src/bboard/*.cpp"
    "deps/pomcpp/src/agents/*.cpp"
    "cpp/src/*.cpp"
    "cpp/include/*.h"
)

include_directories("cpp/include/")
include_directories("deps/pomcpp/include")

find_package (Threads)

add_executable(${PROJECT_NAME} ${source_files})

target_link_libraries( ${PROJECT_NAME} ${CMAKE_THREAD_LIBS_INIT} )
