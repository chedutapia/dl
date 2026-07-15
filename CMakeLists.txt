cmake_minimum_required(VERSION 3.16)
project(deep_learning LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Torch REQUIRED)

add_subdirectory(prueba0)
add_subdirectory(prueba1)
add_subdirectory(prueba2)
add_subdirectory(prueba3)
add_subdirectory(backprop0)
