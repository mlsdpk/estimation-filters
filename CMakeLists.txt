cmake_minimum_required(VERSION 3.6.0)
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/project-meta-info.in")

project(
    estimation-filters
    VERSION ${project_version}
    DESCRIPTION ${project_description})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build thirdparty libraries
add_subdirectory(thirdparty)