# Minimum CMake version
cmake_minimum_required(VERSION 3.16)

# Project definition
project(KokkosBridge LANGUAGES CXX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Locate the Kokkos installation
set(Kokkos_ROOT "/Users/delahondes/mamba")
list(APPEND CMAKE_PREFIX_PATH "${Kokkos_ROOT}")

# Find the Kokkos package
find_package(Kokkos REQUIRED)

# Use kokkos_launch_compiler as a wrapper for the compiler and linker
set(CMAKE_CXX_COMPILER "/Library/Developer/CommandLineTools/usr/bin/c++") # Standard compiler
set(CMAKE_RULE_LAUNCH_COMPILE "/Users/delahondes/mamba/bin/kokkos_launch_compiler")
set(CMAKE_RULE_LAUNCH_LINK "/Users/delahondes/mamba/bin/kokkos_launch_compiler")

# Add your source files
add_library(kokkos_bridge STATIC kokkos_bridge.cpp)

# Link Kokkos to the bridge library
target_link_libraries(kokkos_bridge Kokkos::kokkos)

# Add include directories
target_include_directories(kokkos_bridge PRIVATE ${Kokkos_INCLUDE_DIRS})
