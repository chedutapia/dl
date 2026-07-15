cmake_minimum_required(VERSION 3.16)
project(backprop0 LANGUAGES CXX)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)

  find_package(Torch REQUIRED)
endif()

add_executable(backprop0 src/main.cpp)
target_link_libraries(backprop0 PRIVATE "${TORCH_LIBRARIES}")
target_compile_options(backprop0 PRIVATE -Wall -Wextra -Wpedantic)
