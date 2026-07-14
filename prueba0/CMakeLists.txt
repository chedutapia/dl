cmake_minimum_required(VERSION 3.16)
project(prueba0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Torch REQUIRED)

add_executable(prueba0 src/main.cpp)
target_link_libraries(prueba0 PRIVATE "${TORCH_LIBRARIES}")
target_compile_options(prueba0 PRIVATE -Wall -Wextra -Wpedantic)
