cmake_minimum_required(VERSION 3.16)
project(KokkosBridge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0")

set(Kokkos_ROOT "/Users/delahondes/mamba")
list(APPEND CMAKE_PREFIX_PATH "${Kokkos_ROOT}")

find_package(Kokkos REQUIRED)

set(CMAKE_RULE_LAUNCH_COMPILE "/Users/delahondes/mamba/bin/kokkos_launch_compiler")
set(CMAKE_RULE_LAUNCH_LINK "/Users/delahondes/mamba/bin/kokkos_launch_compiler")

add_library(kokkos_bridge STATIC kokkos_bridge.cpp)
target_link_libraries(kokkos_bridge Kokkos::kokkos)
target_include_directories(kokkos_bridge PRIVATE ${Kokkos_INCLUDE_DIRS})
#
## Add install targets
install(TARGETS kokkos_bridge
        ARCHIVE DESTINATION ${CMAKE_BINARY_DIR}/lib
        LIBRARY DESTINATION ${CMAKE_BINARY_DIR}/lib
        RUNTIME DESTINATION ${CMAKE_BINARY_DIR}/bin)
install(DIRECTORY ${Kokkos_INCLUDE_DIRS} DESTINATION .)
#add_library(kokkos_bridge STATIC kokko_bridge.cpp)
#install(TARGETS kokkos_bridge DESTINATION .)
