cmake_minimum_required(VERSION 3.21) # Minimal for CMAKE_TOOLCHAIN_FILE
# TODO: Delete the cache (File > Delete cach in CMake GUI) for CMake regeneration

# TODO: Change the vcpkg location if installed at different location
set(VCPKG_ROOT "C:/dev/vcpkg")
set(CMAKE_TOOLCHAIN_FILE ${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake)

# Project paths setup for binary, lib and archive
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/Binaries)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/Archive)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/Libraries)

# Vulkan project paths setup
# TODO: These variables' path have to be relative to the project root for ease
# of management between programmers, except for vcpkg related variables.
set(VULKAN_SDK "${CMAKE_SOURCE_DIR}/Dependencies/VulkanSDK")
set(SHADER_SOURCE "${CMAKE_SOURCE_DIR}/Shader/")
set(SHADER_BINARY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Shader")
set(SHADER_VALIDATOR "${CMAKE_SOURCE_DIR}/Shader/Checksums.txt")
set(SHADER_COMPILER "${VULKAN_SDK}/Bin/glslc.exe")
set(VOLUME_DATA "${CMAKE_SOURCE_DIR}/Dependencies/VolumeData")
# Preprocessor definitions, accessible via C++ macros, NDEBUG is already defined
add_definitions(-DPROJECT_ROOT="${CMAKE_SOURCE_DIR}")
add_definitions(-DSHADER_SOURCE="${SHADER_SOURCE}")
add_definitions(-DSHADER_BINARY="${SHADER_BINARY}")
add_definitions(-DSHADER_VALIDATOR="${SHADER_VALIDATOR}")
add_definitions(-DSHADER_COMPILER="${SHADER_COMPILER}")
add_definitions(-DVOLUME_DATA="${VOLUME_DATA}")

# Add files/dirs
file(MAKE_DIRECTORY ${SHADER_BINARY})
file(TOUCH ${SHADER_VALIDATOR})

project(Vulkan)
set(VS_STARTUP_PROJECT ${PROJECT_NAME}) # Set this project as Startup Project in Visual Studio

# Source files
file(GLOB SOURCES "Source/*")
file(GLOB INCLUDES "Include/*")
file(GLOB SHADERS "Shader/*")
source_group("Source" FILES ${SOURCES})
source_group("Include" FILES ${INCLUDES})
source_group("Shader" FILES ${SHADERS})

# Executable
add_executable(${PROJECT_NAME} ${SOURCES} ${INCLUDES} ${SHADERS})
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_23)
target_include_directories(${PROJECT_NAME} PRIVATE
	"${CMAKE_SOURCE_DIR}/Include"
	"${VULKAN_SDK}/Include")

# Dependencies
find_package(glm CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(cryptopp CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE
	${VULKAN_SDK}/Lib/vulkan-1.lib
	glfw
	glm::glm
	cryptopp::cryptopp)
# target_link_directories






