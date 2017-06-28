file (GLOB SOURCES "${PROJECT_SOURCE_DIR}/util/*.cpp")

# configure compiler
add_compile_options (-Wall)
add_compile_options (-std=c++11)
include_directories ("${PROJECT_SOURCE_DIR}/include" "/opt/X11/include")

# generate the executable
add_executable (deconvrl_util ${SOURCES})
target_link_libraries (deconvrl_util LINK_PUBLIC deconvlr)

# drop the suffix
set_target_properties(deconvrl_util PROPERTIES
    OUTPUT_NAME "deconvrl"
)

# add custom target to run the binary
add_custom_target (run
    COMMAND deconvrl_util
    DEPENDS deconvrl_util
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/bin"
)
