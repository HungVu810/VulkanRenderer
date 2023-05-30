#pragma once
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>
#include "Utilities.h"
#include <iostream>
#include <vector>
#include <functional>
#include <optional>

#ifdef NDEBUG
	const bool isValidationLayersEnabled = false;
#else
	const bool isValidationLayersEnabled = true;
#endif

// Mark constructor as explicit
// If the application need to modifies this Vulkan application, use this RunInfo struct to do so implicitly

namespace
{
	using QueueFamilyIndex = uint32_t;
	using QueuesPriorities = std::vector<float>;
	using QueueFamily = std::pair<QueueFamilyIndex, QueuesPriorities>;
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
	GLFWwindow* window;
	const vk::Instance instance;
	const vk::SurfaceKHR surface;
	const vk::PhysicalDevice physicalDevice;
	const vk::Device device;
	const vk::Queue queue;
	const std::vector<QueueFamily> queueFamilies;
	const vk::SwapchainKHR swapchain;
	const vk::Format surfaceFormat;
	const vk::Extent2D surfaceExtent;
	const vk::CommandBuffer commandBuffer;
};

namespace
{
	using CallbackFunction = std::function<void(const ApplicationInfo&)>;
	using CallbackRenderFunction = std::function<void(const ApplicationInfo&, uint32_t imageIndex, bool isFirstFrame)>;
	using CallbackImguiFunction = std::function<void()>;

	constexpr auto WIDTH = uint32_t{800}; // 800, 1280
	constexpr auto HEIGHT = uint32_t{800}; // 800, 720
	constexpr auto MAX_INFLIGHT_IMAGES = 1; // Ideal 2, the number of images being simultaneously processed by the CPU and the GPU

	// Public helpers
	[[nodiscard]] inline auto checkFormatFeatures(const vk::PhysicalDevice& physicalDevice, vk::Format format, vk::FormatFeatureFlagBits requestedFormatFeatures)
	{
		const auto supportedFormatFeatures = physicalDevice.getFormatProperties(format).optimalTilingFeatures;
		const auto isSupported = supportedFormatFeatures & requestedFormatFeatures;
		if (!isSupported) throw std::runtime_error{"Requested format features are not supported"};
	}
	inline void submitCommandBufferOnceSynced(const vk::Device& device, const vk::Queue& queue, const vk::CommandBuffer& commandBuffer, const std::function<void(const vk::CommandBuffer& commandBuffer)>& commands) // Synced means the host will wait on the device queue to finish it works
	{
		const auto waitFence = device.createFence(vk::FenceCreateInfo{});
		commandBuffer.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
		commands(commandBuffer);
		commandBuffer.end();
		queue.submit(vk::SubmitInfo{{}, {}, commandBuffer}, waitFence);
		std::ignore = device.waitForFences(waitFence, VK_TRUE, std::numeric_limits<uint64_t>::max());
		device.destroy(waitFence);
	}
	inline void checkVkResult(VkResult result) // For C-API
	{
		if (result != VK_SUCCESS) throw std::runtime_error{"Failed to init ImGUI."};
	}
}

struct RunInfo
{
	RunInfo(
		const std::vector<std::string>& extraInstanceExtensions_
		, const std::vector<std::string>& extraDeviceExtensions_
		, const CallbackFunction& preRenderLoop_
		, const CallbackRenderFunction& renderCommands_ 
		, const std::optional<CallbackImguiFunction>& imguiCommands_
		, const CallbackFunction& postRenderLoop_
		, std::string_view windowName_ = "MyWindow"
	) : extraInstanceExtensions{extraInstanceExtensions_}
		, extraDeviceExtensions{extraDeviceExtensions_}
		, preRenderLoop{preRenderLoop_}
		, renderCommands{renderCommands_}
		, imguiCommands{imguiCommands_}
		, postRenderLoop{postRenderLoop_}
		, windowName{windowName_}
	{}

	// TODO: Change to span<string_view>?
	const std::vector<std::string> extraInstanceExtensions;
	const std::vector<std::string> extraDeviceExtensions;
	const CallbackFunction preRenderLoop; // For pipeline setup, framebuffer, layout transition, etc.
	const CallbackRenderFunction renderCommands; // Record each frame's command buffer
	const std::optional<CallbackImguiFunction> imguiCommands; // Imgui commands used for each frame. Just pass nullopt if the user doesn't want to use imgui
	const CallbackFunction postRenderLoop; // Cleanup preRenderLoop()'s pipeline and resources created. This is called before the actual cleanUp()
	const std::string_view windowName;
};

class VulkanApplication
{
public:
	VulkanApplication();

	~VulkanApplication() noexcept;

	void run(const RunInfo& runInfo) noexcept;

private:
	void initWindow(const RunInfo& runInfo);

	void initVulkan();
	void initDispatcher();
	void initInstance();
	void initDebugMessenger();
	void initSurface();
	void initPhysicalDevice();
	void initDevice();
	void initQueue();
	void initSwapChain();
	void initCommandPool();
	void initCommandBuffer();
	void initSyncObjects();

	void initImGui();
	void initImGuiDescriptorPool();
	void initImGuiImageViews();
	void initImGuiRenderPass();
	void initImGuiFrameBuffer();
	void initImGuiCommandPool();
	void initImGuiCommandBuffer();
	void cleanupImGui();

	//void renderLoop(const CallbackRenderFunction& recordRenderingCommands, const ApplicationInfo& applicationInfo, const CallbackImguiFunction& imguiCommands, std::string_view windowName);
	void renderLoop(const RunInfo& runInfo, const ApplicationInfo& applicationInfo);

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
	// RenderLoop synchronization
	vk::CommandPool commandPool;
	std::vector<vk::CommandBuffer> commandBuffers;
	vk::Semaphore isAcquiredImageReadSemaphore;
	vk::Semaphore isImageRenderedSemaphore;
	vk::Fence isCommandBufferExecutedFence;

	vk::RenderPass imguiRenderPass;
	std::vector<vk::ImageView> imguiSwapchainImageViews;
	vk::DescriptorPool imguiDescriptorPool;
	std::vector<vk::Framebuffer> imguiFramebuffers;
	vk::CommandPool imguiCommandPool;
	std::vector<vk::CommandBuffer> imguiCommandBuffers;
};


