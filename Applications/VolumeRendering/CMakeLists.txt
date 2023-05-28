# Variables
set(PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(SHADER_SOURCE "${PROJECT_ROOT}/Shader/")
set(SHADER_VALIDATOR "${PROJECT_ROOT}/Shader/Checksums.txt")
set(VOLUME_DATA "${PROJECT_ROOT}/Dependencies/VolumeData")
file(GLOB SOURCES "${PROJECT_ROOT}/Source/*")
file(GLOB INCLUDES "${PROJECT_ROOT}/Include/*")
file(GLOB SHADERS "${PROJECT_ROOT}/Shader/*")

# Preprocessor definitions
add_definitions(-DPROJECT_ROOT="${PROJECT_ROOT}")
# Shader, will be used by a validator function in the common Shader.h to compile
# all of the shader in this projects
add_definitions(-DSHADER_SOURCE="${SHADER_SOURCE}")
add_definitions(-DSHADER_VALIDATOR="${SHADER_VALIDATOR}")
# Dependencies
add_definitions(-DVOLUME_DATA="${VOLUME_DATA}")

# Add files/dirs
file(TOUCH ${SHADER_VALIDATOR})

# Show files in directories in IDE
source_group("Source" FILES ${SOURCES})
source_group("Include" FILES ${INCLUDES})
source_group("Shader" FILES ${SHADERS})
source_group("Common Source" FILES ${COMMON_SOURCES})
source_group("Common Include" FILES ${COMMON_INCLUDES})

# Project
add_executable(VolumeRendering ${SOURCES} ${COMMON_SOURCES} ${INCLUDES} ${COMMON_INCLUDES} ${SHADERS})

target_compile_features(VolumeRendering PRIVATE cxx_std_23)

target_include_directories(VolumeRendering PRIVATE
    "${SOLUTION_ROOT}/Include"
    "${PROJECT_ROOT}/Include"
    "${VULKAN_SDK}/Include")
# target_link_directories

# Libs/Dependencies
# The vulkan package does not provide direct Visual Studio integration.
# For manual integration, add $(VULKAN_SDK)/include to your include path.
# https://cmake.org/cmake/help/latest/module/FindVulkan.html
# find_package(Vulkan REQUIRED)
# find_package(VulkanHeaders CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(cryptopp CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
target_link_libraries(VolumeRendering PRIVATE
    # Vulkan::Vulkan
    # Vulkan::Headers
    # Vulkan::glslc
    "${VULKAN_SDK}/Lib/vulkan-1.lib"
    glfw
    glm::glm
    cryptopp::cryptopp
    imgui::imgui)

