cmake_minimum_required(VERSION 3.25)
project(CUDA_Roberts_Filter CUDA)

set(CMAKE_CUDA_STANDARD 23)

add_executable(CUDA_Roberts_Filter main.cu)

set_target_properties(CUDA_Roberts_Filter PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_subdirectory(lib)
add_subdirectory(res)