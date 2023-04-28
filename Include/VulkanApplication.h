#pragma once
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>
#include <iostream>
#include <vector>
#include <thread>
#include <unordered_map>
#include "Utilities.h"
#include "Shader.h"

#ifdef NDEBUG
	const bool bEnableValidationLayers = false;
#else
	const bool isValidationLayersEnabled = true;
#endif

constexpr auto WIDTH = uint32_t{800};
constexpr auto HEIGHT = uint32_t{600};
// constexpr auto MAX_INFLIGHT_IMAGES = 2; // The swapchain support at least 2 presentable images

// Volume data specification
constexpr auto NUM_SLIDES = 113;
constexpr auto SLIDE_HEIGHT = 256;
constexpr auto SLIDE_WIDTH = 256;
constexpr auto NUM_INTENSITY = NUM_SLIDES * SLIDE_HEIGHT * SLIDE_WIDTH;

namespace
{
	using Intensity = uint16_t;

	using QueueFamilyIndex = uint32_t;
	using QueuesPriorities = std::vector<float>;
	using QueueFamily = std::pair<QueueFamilyIndex, QueuesPriorities>;
}

class VulkanApplication
{
public:
	VulkanApplication();

	~VulkanApplication() noexcept;

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
	void initComputePipeline();
	void initVulkan();
	void initFrameBuffer();
	void initCommandPool();
	void initCommandBuffer();
	void initSyncObjects();
	void recordCommandBuffer(vk::CommandBuffer& commandBuffer, uint32_t imageIndex);

	void mainLoop();

	void drawFrame();

	void cleanUp();

	static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
		VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
		VkDebugUtilsMessageTypeFlagsEXT messageType,
		const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
		void* pUserData)
	{
		const auto message = std::string{ pCallbackData->pMessage };
		const auto posError = message.find("Error");
		assertm(posError == std::string::npos, message); // This is likely a programming error
		const auto posWarning = message.find("Warning");
		const auto tagToUse = posWarning == std::string::npos ? tag::validation : tag::warning;
		std::cerr << tagToUse << message << '\n';
		return VK_FALSE;
	}

private:
	std::thread importVolumeDataWorker;
	std::vector<Intensity> intensities; // z-y-x order, contains CT slides

	GLFWwindow* window;
	std::unordered_map<std::string, Shader> shaderMap;
	std::thread validateShadersWorker;
	vk::Instance instance;
	vk::DebugUtilsMessengerCreateInfoEXT debugMessengerCreateInfo; // TODO: Needed ?
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
	vk::Buffer vertexBuffer;
	vk::DeviceMemory vertexBufferMemory;
	vk::PipelineLayout pipelineLayout;
	vk::Pipeline graphicPipeline;
	std::vector<vk::Framebuffer> framebuffers;
	vk::CommandPool commandPool;
	std::vector<vk::CommandBuffer> commandBuffers;
	vk::Semaphore isAcquiredImageRead;
	vk::Semaphore isImageRendered;
	vk::Fence isCommandBufferExecuted;
};


