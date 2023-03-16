#pragma once
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>
#include <iostream>
#include <vector>

#ifdef NDEBUG
	const bool bEnableValidationLayers = false;
#else
	const bool isValidationLayersEnabled = true;
#endif

// Use (void) to silence unused warnings.
#define assertm(exp, msg) assert(((void)msg, exp))

constexpr auto WIDTH = uint32_t{ 800 };
constexpr auto HEIGHT = uint32_t{ 600 };
constexpr auto print = [](const auto& in) { std::cout << in << '\n'; };

namespace tag
{
	constexpr auto warning = std::string_view{    "[[-------WARNING------]] " };
	constexpr auto exception = std::string_view{  "[[------EXCEPTION-----]] " };
	constexpr auto validation = std::string_view{ "[[     VALIDATION     ]] " };
}

class VulkanApplication
{
public:
	VulkanApplication();
	~VulkanApplication();

	// All exceptions are handled in this function so we can clean up the
	// resources thereafter.
	void run() noexcept;

private:
	void initWindow();

	void initDispatcher();
	void initInstance();
	void initDebugMessenger();
	void initSurface();
	void initPhysicalDevice();
	void initDevice();
	void initQueue();
	void initSwapChain();
	void initImageViews();
	void initGraphicPipeline();
	void initVulkan();

	void mainLoop();

	static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
		VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
		VkDebugUtilsMessageTypeFlagsEXT messageType,
		const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
		void* pUserData)
	{
		std::cerr << tag::validation << pCallbackData->pMessage << '\n';
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
	vk::SwapchainKHR swapchain;
	std::vector<vk::ImageView> imageViews;
};




