cmake_minimum_required (VERSION 3.6)

project (deconvlr)
set (CMAKE_BUILD_TYPE Debug)

# set output directories
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/lib")

# find source files
file (GLOB SOURCES "${PROJECT_SOURCE_DIR}/src/*")

# boost
find_package (Boost 1.59.0 COMPONENTS filesystem iostreams REQUIRED)
if (NOT Boost_FOUND)
    message (FATAL_ERROR "Fatal error: Boost (version >= 1.59) required.")
else ()
    message (STATUS "Setting up Boost")
    message (STATUS " Includes - ${Boost_INCLUDE_DIRS}")
    message (STATUS " Library  - ${Boost_LIBRARY_DIRS}")
endif (NOT Boost_FOUND)
# configure boost
set (Boost_USE_STATIC_LIBS OFF)
include_directories (${Boost_INCLUDE_DIRS})
link_directories (${Boost_LIBRARY_DIRS})

# cuda
find_package (CUDA REQUIRED 7.5)
if (NOT CUDA_FOUND)
    message (FATAL_ERROR "Fatal error: CUDA SDK (version >= 7.5) required.")
endif (NOT CUDA_FOUND)
# configure cuda
list (APPEND CUDA_NVCC_FLAGS "-arch=compute_20; -code=sm_20; -code=sm_30; -std=c++11")
cuda_include_directories ("${PROJECT_SOURCE_DIR}/include")

# generate the shared library
cuda_add_library (deconvlr SHARED ${SOURCES})
target_link_libraries (deconvlr ${Boost_LIBRARIES} ${CUDA_LIBRARIES})

# build demo program
add_subdirectory (test)
