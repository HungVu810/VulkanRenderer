# VulkanRenderer
This is an experimental rendering framework using Vulkan in which I implement some of my projects.

# Applications
## Volume Rendering
Visualizer for CT-scanned data. ImGui is used for interacting with the volume grid via adding control points and rotations.
### Build
- Download vcpkg and install the following dependencies: glfw, glm, cryptopp, imgui
- Make sure the compiler support C++23 latest features such as ranges::zip, enumerate
- Build the Vulkan solution via CMake using the root CMakeLists.txt and build the VolumeRendering project
### Controls
- Left mouse to add/move control points.
- Right mouse to change a control point's color or delete a control point.
### Images
![Screenshot 2023-07-09 220641](https://github.com/HungVu810/VulkanRenderer/assets/63895487/1a62af69-f5d5-482c-a317-5241bdb643e4)
![Screenshot 2023-07-07 205747](https://github.com/HungVu810/VulkanRenderer/assets/63895487/817cea82-becd-404a-8f42-4c192197ca4f)
![Screenshot 2023-07-09 221835](https://github.com/HungVu810/VulkanRenderer/assets/63895487/9e943678-f5f5-4436-bff9-9a345e93e414)
![Screenshot 2023-06-14 200216](https://github.com/HungVu810/VulkanRenderer/assets/63895487/2dec7041-a631-4adf-ad67-47ff2bccce4e)
![Screenshot 2023-06-14 200727](https://github.com/HungVu810/VulkanRenderer/assets/63895487/48e7760e-1c6f-4011-9d74-f83690bb8f80)

## Triangle
Displays a basic triangle used for testing the framework, is not functional at the moment because of reorganizations.
