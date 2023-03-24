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

constexpr auto WIDTH = uint32_t{ 800 };
constexpr auto HEIGHT = uint32_t{ 600 };
constexpr auto print = [](const auto& in) { std::cout << in << '\n'; };

namespace tag
{
	constexpr auto warning = std::string_view{    "[[---WARNING---]] " };
	constexpr auto exception = std::string_view{  "[[--EXCEPTION--]] " };
	constexpr auto error     = std::string_view{  "[[----ERROR----]] " };
	constexpr auto validation = std::string_view{ "[[  VALIDATES  ]] " };
}

inline void assertm(bool condition, std::string_view message)
{
	if (!condition)
	{
		std::cerr << tag::error << message << '\n';
		assert(false);
	}
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
	void initRenderPass();
	void initGraphicPipeline();
	void initVulkan();
	void initFrameBuffer();
	void initCommandPool();
	void initCommandBuffer();
	void initSyncObjects();
	void recordCommandBuffer(vk::CommandBuffer& commandBuffer, uint32_t imageIndex);

	void mainLoop();

	void drawFrame();

	static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
		VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
		VkDebugUtilsMessageTypeFlagsEXT messageType,
		const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
		void* pUserData)
	{
		const auto message = std::string{ pCallbackData->pMessage };
		const auto pos = message.find("Error");
		assertm(pos == std::string::npos, message); // This is likely a programming error
		std::cerr << tag::validation << message << '\n';
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
	vk::SurfaceFormatKHR surfaceFormat;
	vk::Extent2D surfaceExtent;
	vk::SwapchainKHR swapchain;
	std::vector<vk::ImageView> swapchainImageViews;
	vk::RenderPass renderPass;
	vk::PipelineLayout pipelineLayout;
	vk::Pipeline graphicPipeline;
	std::vector<vk::Framebuffer> swapchainFramebuffers;
	vk::CommandPool commandPool;
	std::vector<vk::CommandBuffer> commandBuffers;
	vk::Semaphore isFramebufferPrepaired;
	vk::Semaphore isFramebufferRendered;
	vk::Fence isPreviousFramebufferPresented;
};


