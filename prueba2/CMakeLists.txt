cmake_minimum_required(VERSION 3.16)
project(prueba2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Torch REQUIRED)

add_executable(prueba2 src/main.cpp)
target_link_libraries(prueba2 PRIVATE "${TORCH_LIBRARIES}")
target_compile_options(prueba2 PRIVATE -Wall -Wextra -Wpedantic)
