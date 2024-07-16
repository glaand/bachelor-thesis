cmake_minimum_required(VERSION 3.25)

set(CMAKE_C_COMPILER /usr/bin/gcc)
set(CMAKE_CXX_COMPILER /usr/bin/g++)
set(CMAKE_LINKER /usr/bin/ld)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE Debug)
set(
  CMAKE_CXX_FLAGS
  "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -Wno-register -Wno-comment -fPIC \
  -march=native -Wfatal-errors -fvisibility=hidden -fno-builtin -fno-math-errno -fno-trapping-math -fopenmp"
)
set(CMAKE_CTEST_ARGUMENTS "--output-on-failure")
set(CMAKE_CXX11_ABI 0)

project(ModSims VERSION 1.0.0 LANGUAGES CXX)

# Set the binary and runtime output directories
set(CMAKE_BINARY_DIR "${CMAKE_SOURCE_DIR}/cpp/build")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/../out/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/../out/lib")

enable_testing()

# Add extern projects

# Torch
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
list(APPEND CMAKE_PREFIX_PATH "$ENV{HOME}/miniconda3/envs/bachelor-thesis/lib/python3.11/site-packages/torch/share/cmake/Torch")
set(Torch_DIR "$ENV{HOME}/miniconda3/envs/bachelor-thesis/lib/python3.11/site-packages/torch")
set(CMAKE_CUDA_ARCHITECTURES "native")

# Function to get hostname
function(get_hostname result)
    execute_process(
        COMMAND hostname
        OUTPUT_VARIABLE ${result}
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endfunction()

# Get hostname
get_hostname(HOSTNAME)

# Set CUDA compiler based on hostname
if (HOSTNAME STREQUAL "mercury")
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.5/bin/nvcc")
else()
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.3/bin/nvcc")
endif()

# VTK
set(VTK_GROUP_ENABLE_StandAlone DONT_WANT)
set(VTK_GROUP_ENABLE_Rendering DONT_WANT)
set(VTK_MODULE_ENABLE_VTK_CommonDataModel YES)
set(VTK_MODULE_ENABLE_VTK_CommonCore YES)
set(VTK_MODULE_ENABLE_VTK_CommonExecutionModel YES)
set(VTK_MODULE_ENABLE_VTK_CommonMath YES)
set(VTK_MODULE_ENABLE_VTK_CommonMisc YES)
set(VTK_MODULE_ENABLE_VTK_CommonSystem YES)
set(VTK_MODULE_ENABLE_VTK_CommonTransforms YES)
set(VTK_MODULE_ENABLE_VTK_IOCore YES)
set(VTK_MODULE_ENABLE_VTK_IOLegacy YES)
set(VTK_MODULE_ENABLE_VTK_IOParallelXML YES)
set(VTK_MODULE_ENABLE_VTK_IOXML YES)
set(VTK_MODULE_ENABLE_VTK_IOXMLParser YES)
set(VTK_MODULE_ENABLE_VTK_ParallelCore YES)
set(VTK_MODULE_ENABLE_VTK_FiltersCore YES)
set(VTK_MODULE_ENABLE_VTK_vtksysYES)
add_subdirectory(extern/vtk)

# Include libraries
add_subdirectory(kernel)
add_subdirectory(cfd_2d)
add_subdirectory(cfd_3d)