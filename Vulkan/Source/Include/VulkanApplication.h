#pragma once
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
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

// Use (void) to silence unused warnings.
#define assertm(exp, msg) assert(((void)msg, exp))

constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;
constexpr auto print = [](const auto& in) { std::cout << in << '\n'; };

#ifdef NDEBUG
	const bool enableValidationLayers = false;
#else
	const bool enableValidationLayers = true;
#endif

class VulkanApplication
{
public:
	VulkanApplication();
	// All exceptions are handled in this function so we can clean up the
	// resources thereafter.
	auto run() noexcept -> void;

private:
	auto initWindow() -> void;

	auto isValidationLayerSupported() const noexcept -> bool;
	auto setRequiredLayerNames() noexcept -> void;
	auto setRequiredExtensionNames() -> std::vector<const char*>;
	auto setInstanceCreateInfo() -> void;
	auto createVulkanInstance() -> void;
	auto initVulkan() -> void;

	auto mainLoop() -> void;

	auto cleanup() -> void;

	static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
		VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
		VkDebugUtilsMessageTypeFlagsEXT messageType,
		const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
		void* pUserData)
	{
		std::cerr << "Validation Layer: " << pCallbackData->pMessage << '\n';
		return VK_FALSE;
	}

	GLFWwindow* window;
	vk::Instance instance;
	vk::ApplicationInfo applicationInfo;
	std::vector<const char*> requiredExtensionNames;
	std::vector<const char*> requiredLayerNames;
	vk::InstanceCreateInfo instanceCreateInfo;
};




