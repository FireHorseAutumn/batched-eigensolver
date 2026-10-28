cmake_minimum_required(VERSION 3.20)

# Select a supported CUDA host compiler before project() enables CUDA.
# With GCC 16 or newer, prefer g++-15; otherwise enable the compiler override
# and use C++20 for CUDA, since newer libstdc++ headers require it.
find_program(_cxx_compiler NAMES $ENV{CXX} c++ g++)
if(_cxx_compiler)
    execute_process(COMMAND ${_cxx_compiler} -dumpversion
                    OUTPUT_VARIABLE _gcc_version OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET)
endif()
if(_gcc_version VERSION_GREATER_EQUAL 16)
    find_program(_gcc15 NAMES g++-15)
    if(_gcc15)
        set(CMAKE_CUDA_HOST_COMPILER "${_gcc15}")
    else()
        set(CMAKE_CUDA_FLAGS_INIT "-allow-unsupported-compiler -std=c++20")
    endif()
endif()

project(batched_eigensolver LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(CMAKE_CUDA_HOST_COMPILER)
    set(CMAKE_CUDA_STANDARD 17)
else()
    set(CMAKE_CUDA_STANDARD 20)
endif()
set(CMAKE_CUDA_ARCHITECTURES native)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Eigen is the oracle only (test_eig). The solver in include/beig is
# dependency-free.
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(CUDAToolkit REQUIRED)

enable_testing()

file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/data)

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

add_executable(test_gpu_eig tests/test_gpu_eig.cu)
target_link_libraries(test_gpu_eig PRIVATE beig CUDA::cudart)
target_include_directories(test_gpu_eig PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_compile_options(test_gpu_eig PRIVATE -Wall -Wextra)
add_test(NAME gpu COMMAND test_gpu_eig)

string(TOUPPER "${CMAKE_BUILD_TYPE}" _build_type_upper)

add_executable(bench_eig benchmarks/bench_eig.cu)
target_link_libraries(bench_eig PRIVATE beig CUDA::cudart CUDA::cusolver)
target_include_directories(bench_eig PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_compile_options(bench_eig PRIVATE -Wall -Wextra)
target_compile_definitions(bench_eig PRIVATE
  BEIG_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
  BEIG_BUILD_TYPE="${CMAKE_BUILD_TYPE}")

add_executable(bench_phases benchmarks/bench_phases.cu)
target_link_libraries(bench_phases PRIVATE beig CUDA::cudart)
target_include_directories(bench_phases PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_compile_options(bench_phases PRIVATE -Wall -Wextra)
