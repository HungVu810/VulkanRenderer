#pragma once
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>
#include <ranges>
#include <iostream>
#include <stdexcept>
#include <cassert>
#include <cstdlib>
#include <memory>
#include <vector>
#include <algorithm>
#include <cstring>
#include <array>
#include <functional>
#include <cstdint> // Needed for uint32_t
#include <limits> // Needed for std::numeric_limits
#include <algorithm> // Needed for std::clamp

#ifdef NDEBUG
	const bool enableValidationLayers = false;
#else
	const bool enableValidationLayers = true;
#endif

// Use (void) to silence unused warnings.
#define assertm(exp, msg) assert(((void)msg, exp))

constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;
constexpr auto print = [](const auto& in) { std::cout << in << '\n'; };

// Forward declare --------- START
namespace
{
	class Names;
	struct SurfaceAttributes;
}
// Forward declare --------- END

class VulkanApplication
{
public:
	VulkanApplication();
	~VulkanApplication();

	// All exceptions are handled in this function so we can clean up the
	// resources thereafter.
	auto run() noexcept -> void;

private:
	auto initWindow() -> void;

	auto initDispatcher() -> void;
	auto initInstance() -> void;
	auto initDebugMessenger() -> void;
	auto initSurface() -> void;
	auto initPhysicalDevice() -> void;
	auto initDevice() -> void;
	auto initQueue() -> void;
	auto initSwapChain() -> void;
	auto initVulkan() -> void;

	auto mainLoop() -> void;

	static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
		VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
		VkDebugUtilsMessageTypeFlagsEXT messageType,
		const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
		void* pUserData)
	{
		std::cerr << "[[ VALIDATION LAYER ]] " << pCallbackData->pMessage << '\n';
		return VK_FALSE;
	}

private:
	GLFWwindow* window;
	vk::Instance instance;
	vk::DebugUtilsMessengerCreateInfoEXT debugMessengerCreateInfo; // needed ?
	vk::DebugUtilsMessengerEXT debugMessenger;
	vk::SurfaceKHR surface;
	vk::PhysicalDevice physicalDevice;
	vk::Device device;
	vk::Queue queue;
	vk::SwapchainKHR swapChain;
};




