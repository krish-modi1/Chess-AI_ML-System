cmake_minimum_required(VERSION 3.12)
project(mcts_engine_cpp)

# Use C++17 for structured bindings and other modern features
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

# Find Python and pybind11
# First try finding pybind11 via CMake config
execute_process(
    COMMAND python -c "import pybind11; import os; print(os.path.dirname(pybind11.__file__))"
    OUTPUT_VARIABLE PYBIND11_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(PYBIND11_PATH)
    set(pybind11_DIR "${PYBIND11_PATH}/share/cmake/pybind11")
    message(STATUS "Found pybind11 at: ${pybind11_DIR}")
endif()

# Find required packages
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Create the shared library module
pybind11_add_module(mcts_engine_cpp
    src/mcts_engine.cpp
    src/python_bridge.cpp
)

# Add source directory to include path
target_include_directories(mcts_engine_cpp PRIVATE src)

# Compiler flags for optimization
if(MSVC)
    target_compile_options(mcts_engine_cpp PRIVATE /O2)
else()
    target_compile_options(mcts_engine_cpp PRIVATE -O3 -march=native -ffast-math)
endif()

# Installation
install(TARGETS mcts_engine_cpp LIBRARY DESTINATION game_engine)

message(STATUS "MCTS C++ Engine Build Configuration:")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Python: ${Python3_EXECUTABLE}")
message(STATUS "  pybind11: ${pybind11_DIR}")
message(STATUS "  Source files: src/mcts_engine.cpp, src/python_bridge.cpp")