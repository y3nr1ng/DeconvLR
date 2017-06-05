file (GLOB SOURCES "${PROJECT_SOURCE_DIR}/util/*.cpp")

# configure compiler
add_compile_options (-Wall)
add_compile_options (-std=c++14)
include_directories ("${PROJECT_SOURCE_DIR}/include" "/opt/X11/include")

# generate the executable
add_executable (deconvlr_util ${SOURCES})
target_link_libraries (deconvlr_util LINK_PUBLIC deconvlr)

# drop the suffix
set_target_properties(deconvlr_util PROPERTIES
    OUTPUT_NAME "deconvlr"
)

# add custom target to run the binary
add_custom_target (run
    COMMAND deconvlr_util
    DEPENDS deconvlr_util
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/bin"
)
