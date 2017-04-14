cmake_minimum_required (VERSION 3.6)

project (deconvlr)
set (CMAKE_BUILD_TYPE Debug)

# set output directories
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/lib")

# headers
include_directories (include)
# find source files
file(GLOB SOURCES "${PROJECT_SOURCE_DIR}/src/*")

# configure boost
set (Boost_USE_STATIC_LIBS OFF)
find_package (Boost 1.59.0 COMPONENTS filesystem iostreams REQUIRED)
# test the existence
if (NOT Boost_FOUND)
    message(FATAL_ERROR "Fatal error: Boost (version >= 1.59) required.")
else ()
    message(STATUS "Setting up Boost")
    message(STATUS " Includes - ${Boost_INCLUDE_DIRS}")
    message(STATUS " Library  - ${Boost_LIBRARY_DIRS}")
    include_directories(${Boost_INCLUDE_DIRS})
    link_directories(${Boost_LIBRARY_DIRS})
endif (NOT Boost_FOUND)

# configure cuda
find_package (CUDA REQUIRED)
# nvcc flags
list (APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_20,code=sm_20")

# configure compiler flag
add_compile_options(-Wall)
add_compile_options(-std=c++11)

# generate the shared library
add_library(deconvlr SHARED ${SOURCES})
target_link_libraries(deconvlr ${Boost_LIBRARIES})

# compile and link
cuda_add_executable(demo ${PROJECT_SOURCE_DIR}/src/DeconvLRDriver.cpp ${CU_O})
target_link_libraries(demo deconvlr)
