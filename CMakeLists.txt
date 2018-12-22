cmake_minimum_required(VERSION 3.2)
project(simpleNN LANGUAGES CXX VERSION 0.1 DESCRIPTION "A simple neural network library")

find_package ( Threads REQUIRED )

set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_COMPILER g++-7)
set(THREADS_PREFER_PTHREAD_FLAG ON)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -g -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

add_executable(simpleNN main.cpp neuralnetwork.cpp neuralnetwork.h)