#pragma once
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>
#include "Utilities.h"
#include "glm/vec2.hpp"
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
	ApplicationInfo(
		GLFWwindow* window_
		, const vk::Instance& instance_
		, const vk::SurfaceKHR& surface_
		, const vk::PhysicalDevice& physicalDevice_
		, const vk::Device& device_
		, const vk::Queue& queue_
		, const std::vector<QueueFamily>& queueFamilies_
		, const vk::SwapchainKHR& swapchain_
		, const vk::Format& surfaceFormat_
		, const vk::Extent2D& surfaceExtent_
		, const vk::CommandBuffer& commandBuffer_
	) : window{window_}
		, instance{instance_}
		, surface{surface_}
		, physicalDevice{physicalDevice_}
		, device{device_}
		, queue{queue_}
		, queueFamilies{queueFamilies_}
		, swapchain{swapchain_}
		, surfaceFormat{surfaceFormat_}
		, surfaceExtent{surfaceExtent_}
		, commandBuffer{commandBuffer_}
	{}

	GLFWwindow* window;
	vk::Instance instance;
	vk::SurfaceKHR surface;
	vk::PhysicalDevice physicalDevice;
	vk::Device device;
	vk::Queue queue;
	std::vector<QueueFamily> queueFamilies;
	vk::SwapchainKHR swapchain;
	vk::Format surfaceFormat;
	vk::Extent2D surfaceExtent;
	vk::CommandBuffer commandBuffer;

	ApplicationInfo& operator=(const ApplicationInfo& applicationInfo) noexcept
	{
		window = applicationInfo.window;
		instance = applicationInfo.instance;
		surface = applicationInfo.surface;
		physicalDevice = applicationInfo.physicalDevice;
		device = applicationInfo.device;
		queue = applicationInfo.queue;
		queueFamilies = applicationInfo.queueFamilies;
		swapchain = applicationInfo.swapchain;
		surfaceFormat = applicationInfo.surfaceFormat;
		surfaceExtent = applicationInfo.surfaceExtent;
		commandBuffer = applicationInfo.commandBuffer;
		return *this;
	}
};

namespace
{
	using CallbackFunction = std::function<void(const ApplicationInfo&)>;
	using CallbackRenderFunction = std::function<void(const ApplicationInfo&, uint32_t imageIndex, bool isFirstFrame)>;
	using CallbackImguiFunction = std::function<void()>;

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

/*
	renderCommands must transition the final image layout to
	colorAttachmentOptimal regardless of whether imgui is used or not since the
	imgui renderpass will transition that image to presentSrcKHR from
	colorAttachmentOptimal
*/
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
		, const glm::vec2& windowExtent_ = glm::vec2{800, 800}
	) : extraInstanceExtensions{extraInstanceExtensions_}
		, extraDeviceExtensions{extraDeviceExtensions_}
		, preRenderLoop{preRenderLoop_}
		, renderCommands{renderCommands_}
		, imguiCommands{imguiCommands_}
		, postRenderLoop{postRenderLoop_}
		, windowName{windowName_}
		, windowExtent{windowExtent_}
	{}

	// TODO: Change to span<string_view>?
	const std::vector<std::string> extraInstanceExtensions;
	const std::vector<std::string> extraDeviceExtensions;
	const CallbackFunction preRenderLoop; // For pipeline setup, framebuffer, layout transition, etc.
	const CallbackRenderFunction renderCommands; // Record each frame's command buffer
	const std::optional<CallbackImguiFunction> imguiCommands; // Imgui commands used for each frame. Just pass nullopt if the user doesn't want to use imgui
	const CallbackFunction postRenderLoop; // Cleanup preRenderLoop()'s pipeline and resources created. This is called before the actual cleanUp()
	const std::string_view windowName;
	const glm::vec2 windowExtent;
};

class VulkanApplication
{
public:
	VulkanApplication();

	~VulkanApplication() noexcept;

	inline const ApplicationInfo& getApplicationInfo() const noexcept;

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
	std::optional<ApplicationInfo> applicationInfo;
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

	// ImGui
	vk::RenderPass imguiRenderPass;
	std::vector<vk::ImageView> imguiSwapchainImageViews;
	vk::DescriptorPool imguiDescriptorPool;
	std::vector<vk::Framebuffer> imguiFramebuffers;
	vk::CommandPool imguiCommandPool;
	std::vector<vk::CommandBuffer> imguiCommandBuffers;
};


