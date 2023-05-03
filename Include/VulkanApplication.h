#pragma once
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>
#include <iostream>
#include <vector>
#include <functional>
#include "Utilities.h"

#ifdef NDEBUG
	const bool isValidationLayersEnabled = false;
#else
	const bool isValidationLayersEnabled = true;
#endif

namespace
{
	using QueueFamilyIndex = uint32_t;
	using QueuesPriorities = std::vector<float>;
	using QueueFamily = std::pair<QueueFamilyIndex, QueuesPriorities>;


	constexpr auto WIDTH = uint32_t{800}; // 800, 1280
	constexpr auto HEIGHT = uint32_t{800}; // 800, 720
	// constexpr auto MAX_INFLIGHT_IMAGES = 2; // The swapchain support at least 2 presentable images

	// Public helpers
	[[nodiscard]] auto inline getQueueFamilyIndices(const std::vector<QueueFamily>& queueFamilies)
	{
		const auto toQueueFamilyIndex = [](const QueueFamily& queueFamily){return queueFamily.first;};
		const auto queueFamiliesIndex = queueFamilies | std::views::transform(toQueueFamilyIndex) | std::ranges::to<std::vector>();
		return queueFamiliesIndex;
	}
	[[nodiscard]] auto inline checkFormatFeatures(const vk::PhysicalDevice& physicalDevice, vk::Format format, vk::FormatFeatureFlagBits requestedFormatFeatures)
	{
		const auto supportedFormatFeatures = physicalDevice.getFormatProperties(format).optimalTilingFeatures;
		const auto isSupported = supportedFormatFeatures & requestedFormatFeatures;
		if (!isSupported) throw std::runtime_error{"Requested format features are not supported"};
	}
}

template <typename T>
struct Resource
{
	T data;
	vk::Format format; // Data type for the underlying elements of data
};
struct SurfaceInfo
{
	vk::Format format;
	vk::Extent2D extent;
};
struct SyncObjects
{
	vk::Semaphore isAcquiredImageReadSemaphore;
	vk::Semaphore isImageRenderedSemaphore;
	vk::Fence isCommandBufferExecutedFence;
};
struct ApplicationInfo
{
	const vk::SurfaceKHR surface;
	const vk::PhysicalDevice physicalDevice;
	const vk::Device device;
	const vk::Queue queue;
	const std::vector<QueueFamily> queueFamilies; // getSuitableQueueFamilies(physicalDevice, surface);
	const vk::SwapchainKHR swapchain;
	const vk::Format surfaceFormat;
	const vk::Extent2D surfaceExtent;
	const std::vector<vk::CommandBuffer> commandBuffers;
	const vk::Semaphore isAcquiredImageReadSemaphore;
	const vk::Semaphore isImageRenderedSemaphore;
	const vk::Fence isCommandBufferExecutedFence;
};

namespace
{
	using ApplicationFunction = std::function<void(const ApplicationInfo&)>;
}

struct RunInfo
{
	RunInfo(
		const std::vector<std::string>& inExtraInstanceExtensions
		, const std::vector<std::string>& inExtraDeviceExtensions
		, vk::ImageUsageFlagBits inSwapchainImageUsage
		, const ApplicationFunction& inPreRenderLoop
		, const ApplicationFunction& inRenderFrame
		, const ApplicationFunction& inPostRenderLoop
	) : extraInstanceExtensions{inExtraInstanceExtensions}
		, extraDeviceExtensions{inExtraDeviceExtensions}
		, swapchainImageUsage{inSwapchainImageUsage}
		, preRenderLoop{inPreRenderLoop}
		, renderFrame{inRenderFrame}
		, postRenderLoop{inPostRenderLoop}
	{}

	// TODO: If the Application need to modifies this application, use this RunInfo struct to do so implicitly
	const std::vector<std::string> extraInstanceExtensions;
	const std::vector<std::string> extraDeviceExtensions;
	const vk::ImageUsageFlagBits swapchainImageUsage;
	const ApplicationFunction preRenderLoop; // For pipeline setup, framebuffer, layout transition, etc.
	const ApplicationFunction renderFrame; // Buffering recording and rendering for one frame
	const ApplicationFunction postRenderLoop; // Cleanup of pipeline and resources created via preRenderLoop(). This is called before the actual cleanUp()
	// TODO: Change to span<string_view>?
	// TODO: Make class vulkan a struct so the other main application can interact with via their struct/class
};

class VulkanApplication
{
public:
	VulkanApplication();

	~VulkanApplication() noexcept;

	void run(const RunInfo& runInfo) noexcept;

private:
	void initWindow();

	void initVulkan(const RunInfo& runInfo);
	void initDispatcher();
	void initInstance();
	void initDebugMessenger();
	void initSurface();
	void initPhysicalDevice();
	void initDevice();
	void initQueue();
	void initSwapChain(vk::ImageUsageFlagBits swapchainImageUsage);
	void initCommandPool();
	void initCommandBuffer();
	void initSyncObjects();

	void renderLoop(const ApplicationFunction& renderFrame, const ApplicationInfo& applicationInfo);

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
	GLFWwindow* window;
	vk::Instance instance;
	vk::DebugUtilsMessengerCreateInfoEXT debugMessengerCreateInfo;
	vk::DebugUtilsMessengerEXT debugMessenger;
	vk::SurfaceKHR surface;
	vk::PhysicalDevice physicalDevice;
	vk::Device device;
	vk::Queue queue;
	vk::SurfaceFormatKHR surfaceFormat;
	vk::Extent2D surfaceExtent;
	vk::SwapchainKHR swapchain;
	// RenderLoop synchronizatoin
	vk::CommandPool commandPool;
	std::vector<vk::CommandBuffer> commandBuffers;
	vk::Semaphore isAcquiredImageReadSemaphore;
	vk::Semaphore isImageRenderedSemaphore;
	vk::Fence isCommandBufferExecutedFence;
};


