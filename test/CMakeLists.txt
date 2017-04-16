file (GLOB SOURCES "${PROJECT_SOURCE_DIR}/test/*.cpp")

# configure compiler
add_compile_options (-Wall)
add_compile_options (-std=c++14)
include_directories ("${PROJECT_SOURCE_DIR}/include" "/opt/X11/include")

# generate the executable
add_executable (deconvlr_demo ${SOURCES})
target_link_libraries (deconvlr_demo LINK_PUBLIC deconvlr)

# add custom target to run the binary
add_custom_target (run
    COMMAND deconvlr_demo
    DEPENDS deconvlr_demo
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/bin"
)
