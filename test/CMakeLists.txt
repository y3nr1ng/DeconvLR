file (GLOB SOURCES "${PROJECT_SOURCE_DIR}/test/*.cpp")

# configure compiler
add_compile_options (-Wall)
add_compile_options (-std=c++11)
include_directories ("${PROJECT_SOURCE_DIR}/include")

# generate the executable
add_executable (deconvlr_demo ${SOURCES})
target_link_libraries (deconvlr_demo LINK_PUBLIC deconvlr)
