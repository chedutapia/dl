cmake_minimum_required(VERSION 3.16)
project(prueba1 LANGUAGES CXX)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)

  find_package(Torch REQUIRED)
endif()

add_executable(prueba1 src/main.cpp)
target_link_libraries(prueba1 PRIVATE "${TORCH_LIBRARIES}")
target_compile_options(prueba1 PRIVATE -Wall -Wextra -Wpedantic)
