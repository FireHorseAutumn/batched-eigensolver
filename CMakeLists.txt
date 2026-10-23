cmake_minimum_required(VERSION 3.20)
project(batched_eigensolver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Eigen is the oracle only (test_eig). The solver in include/beig is
# dependency-free.
find_package(Eigen3 REQUIRED NO_MODULE)

enable_testing()

add_library(beig INTERFACE)
target_include_directories(beig INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(test_eig tests/test_eig.cpp)
target_link_libraries(test_eig PRIVATE beig Eigen3::Eigen)
target_compile_options(test_eig PRIVATE -Wall -Wextra)
add_test(NAME eig COMMAND test_eig)

add_executable(test_ribbon tests/test_ribbon.cpp)
target_link_libraries(test_ribbon PRIVATE beig)
target_compile_options(test_ribbon PRIVATE -Wall -Wextra)
add_test(NAME ribbon COMMAND test_ribbon)
