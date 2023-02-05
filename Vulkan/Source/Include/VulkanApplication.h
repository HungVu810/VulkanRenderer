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

class Names // forward declare
{
public:
	Names() : names{}, proxy{} {};
	Names(std::vector<std::string>&& inNames) : names{ inNames }, proxy{} {};
	[[nodiscard]] auto getProxy() noexcept -> const std::vector<const char*>&
	{
		proxy.clear();
		proxy.resize(names.size());
		const auto populateProxy = [this](size_t i)
		{
			proxy[i] = names[i].data();
		};
		std::ranges::for_each(std::views::iota(0u, proxy.size()), populateProxy);
		return proxy;
	}

public:
	std::vector<std::string> names;

private:
	std::vector<const char*> proxy; // Used with the Vulkan API, which is just vector names's std::string casted to const char* const
};

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
	auto initLayer() noexcept -> void;
	auto initInstanceExtension() noexcept -> void;
	auto initInstanceCreateInfo() -> void;
	[[nodiscard]] auto getSuitablePhysicalDevice() const -> vk::PhysicalDevice;
	auto initQueueCreateInfos(const vk::PhysicalDevice& physicalDevice) noexcept -> void;
	auto initDeviceExtension() noexcept -> void;
	auto initDevice() -> void;
	auto initQueue() -> void;
	auto initDebugMessenger() -> void;
	auto initWindowSurface() -> void;
	auto initVulkan() -> void;

	auto mainLoop() -> void;

	auto cleanup() -> void;

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
	vk::ApplicationInfo applicationInfo;
	Names layer, instanceExtension, deviceExtension;
	vk::DebugUtilsMessengerCreateInfoEXT debugMessengerCreateInfo;
	vk::InstanceCreateInfo instanceCreateInfo;
	vk::Instance instance;
	vk::DebugUtilsMessengerEXT debugMessenger;
	// stores a queue family index, and the associated queues + their priority level
	using queueFamilyIndex = uint32_t;
	using queuesPriority = std::vector<float>;
	using queueFamily = std::pair<queueFamilyIndex, queuesPriority>;
	std::vector<queueFamily> queueFamilies;
	std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
	vk::Device device;
	vk::Queue queue;
	vk::SurfaceKHR surface;

};




