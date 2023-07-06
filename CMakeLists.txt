cmake_minimum_required(VERSION 3.21) # Minimal for CMAKE_TOOLCHAIN_FILE
# TODO: Delete the cache (File > Delete cach in CMake GUI) for CMake regeneration

# TODO: Change the vcpkg location if installed at different location
set(VCPKG_ROOT "C:/dev/vcpkg")
set(CMAKE_TOOLCHAIN_FILE ${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake)

# Project paths setup for binary, lib and archive
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/Binaries)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/Archive)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/Libraries)

# Preprocessor definitions, accessible via c++ macros, ndebug is already defined
set(SOLUTION_ROOT "${CMAKE_SOURCE_DIR}")
set(VULKAN_SDK "${SOLUTION_ROOT}/Dependencies/VulkanSDK")
set(SHADER_BINARY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Shader")
set(SHADER_COMPILER "${VULKAN_SDK}/Bin/glslc.exe")
add_definitions(-DSOLUTION_ROOT="${SOLUTION_ROOT}")
add_definitions(-DSHADER_BINARY="${SHADER_BINARY}")
add_definitions(-DSHADER_COMPILER="${SHADER_COMPILER}")

# Add files/dirs
file(MAKE_DIRECTORY ${SHADER_BINARY})

# Source files
file(GLOB COMMON_SOURCES "${SOLUTION_ROOT}/Source/*")
file(GLOB COMMON_INCLUDES "${SOLUTION_ROOT}/Include/*")

# Solution
project(Vulkan)

# Need to set these variable for Imgui's vcpkg dependencies
set(Vulkan_LIBRARY "${VULKAN_SDK}/Lib/vulkan-1.lib")
set(Vulkan_INCLUDE_DIR "${VULKAN_SDK}/Include")
# Expand subdirectory
add_subdirectory(Applications)

# Set a startup project in Visual Studio IDE
set_property(DIRECTORY ${SOLUTION_ROOT} PROPERTY VS_STARTUP_PROJECT VolumeRendering)

# Packaging for distribution


