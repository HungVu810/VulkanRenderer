#include <ranges>
#include <algorithm>
#include <stdexcept>
#include <cstdint> // Needed for uint32_t
#include <limits> // Needed for std::numeric_limits
#include "VulkanApplication.h" // Do not place this right above the VULKAN_HPP_DEFAULT macro
#include "Utilities.h"
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace
{
	// Classes/Structs
	struct SurfaceAttributes
	{
		vk::SurfaceCapabilitiesKHR capabilities;
		std::vector<vk::SurfaceFormatKHR> formats;
		std::vector<vk::PresentModeKHR> presentModes;
	};

	// Private helpers
	[[nodiscard]] auto getNamesProxy(const std::vector<std::string>& names) noexcept // Return a container of cstr proxies from names, which is used by the Vulkan API
	{
		const auto toRawString = [](std::string_view str) { return str.data(); };
		auto proxies = names | std::views::transform(toRawString) | std::ranges::to<std::vector>();
		return proxies;
	}
	[[nodiscard]] auto getEnabledLayersName()
	{
		auto strings = std::vector<std::string>{};
		if (!isValidationLayersEnabled) return strings;
		const auto validationLayerName = std::string{"VK_LAYER_KHRONOS_validation"};
		const auto layersProperties = vk::enumerateInstanceLayerProperties();
		const auto getLayerName = [](const vk::LayerProperties& layerProperties) { return std::string_view(layerProperties.layerName); };
		const auto layersName = layersProperties | std::views::transform(getLayerName);
		const auto isSupported = std::ranges::find(layersName, validationLayerName) != std::end(layersName);
		if (!isSupported) throw std::runtime_error{"Requested validation layers name can't be found."};
		strings.push_back(validationLayerName);
		return strings;
	}
	[[nodiscard]] auto getEnabledInstanceExtensionsName()
	{
		auto extensionCount = uint32_t{ 0 };
		const auto glfwExtensionsRawName = glfwGetRequiredInstanceExtensions(&extensionCount);
		if (!glfwExtensionsRawName) throw std::runtime_error{"Vulkan's surface extensions for window creation can not be found"};
		auto extensionsName = std::vector<std::string>{ glfwExtensionsRawName, glfwExtensionsRawName + extensionCount };
		if (isValidationLayersEnabled) extensionsName.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
		return extensionsName;
	}
	[[nodiscard]] auto getDeviceExtensionsName() noexcept
	{
		return std::vector<std::string>{ VK_KHR_SWAPCHAIN_EXTENSION_NAME };
	}
	[[nodiscard]] auto getSurfaceAttributes(const vk::SurfaceKHR& surface, const vk::PhysicalDevice& physicalDevice) noexcept
	{
		return SurfaceAttributes {
			.capabilities = physicalDevice.getSurfaceCapabilitiesKHR(surface),
			.formats = physicalDevice.getSurfaceFormatsKHR(surface),
			.presentModes = physicalDevice.getSurfacePresentModesKHR(surface)
		};
	}
	[[nodiscard]] auto getSuitablePhysicalDevice(const std::vector<vk::PhysicalDevice>& physicalDevices, const vk::SurfaceKHR& surface)
	{
		const auto deviceExtensionsName = getDeviceExtensionsName();
		const auto isSuitable = [&](const vk::PhysicalDevice& physicalDevice)
		{
			const auto deviceProperties = vk::PhysicalDeviceProperties{physicalDevice.getProperties()};
			const auto deviceFeatures = vk::PhysicalDeviceFeatures{physicalDevice.getFeatures()};
			const auto getExtensionName = [](const vk::ExtensionProperties& properties)
			{
				return static_cast<std::string_view>(properties.extensionName);
			};
			const auto supportedDeviceExtensionsName = physicalDevice.enumerateDeviceExtensionProperties() | std::views::transform(getExtensionName);
			const auto surfaceAttributes = getSurfaceAttributes(surface, physicalDevice);
			const auto isSupported = [&](const std::string_view& deviceExtensionName)
			{
				return std::ranges::find(supportedDeviceExtensionsName, deviceExtensionName) != supportedDeviceExtensionsName.end();
			};
			return deviceFeatures.geometryShader
				&& std::ranges::all_of(deviceExtensionsName, isSupported)
				&& !surfaceAttributes.formats.empty()
				&& !surfaceAttributes.presentModes.empty();
		};
		auto suitablePhysicalDevices = physicalDevices | std::views::filter(isSuitable);
		if (suitablePhysicalDevices.empty()) throw std::runtime_error{ "Can't find a suitable GPU." };
		const auto isDiscrete = [](const vk::PhysicalDevice& physicalDevice)
		{
			return physicalDevice.getProperties().deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
		};
		const auto discretePhysicalDeviceIter = std::ranges::find_if(suitablePhysicalDevices, isDiscrete);
		if (discretePhysicalDeviceIter == suitablePhysicalDevices.end())
		{
			std::cout << tag::warning << "Using a non-discrete device for rendering.\n";
			return suitablePhysicalDevices.front();
		}
		else return *discretePhysicalDeviceIter;
	}
	[[nodiscard]] auto getSuitableSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& surfaceFormats) noexcept
	{
		const auto isSuitable = [](const vk::SurfaceFormatKHR& surfaceFormat)
		{
			return surfaceFormat.format == format::Image
				&& surfaceFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear;
		};
		const auto surfaceFormatIter = std::ranges::find_if(surfaceFormats, isSuitable);
		if (surfaceFormatIter == surfaceFormats.end()) return surfaceFormats.at(0);
		else return *surfaceFormatIter;
	}
	[[nodiscard]] auto getSuitablePresentMode(const std::vector<vk::PresentModeKHR>& presentModes) noexcept
	{
		const auto isSuitable = [](const vk::PresentModeKHR& presentMode) { return presentMode == vk::PresentModeKHR::eMailbox; };
		const auto presentModesIter = std::ranges::find_if(presentModes, isSuitable);
		if (presentModesIter == presentModes.end()) return vk::PresentModeKHR::eFifo;
		else return *presentModesIter;
	}
	[[nodiscard]] auto getSuitableSurfaceExtent(GLFWwindow* window, const vk::SurfaceCapabilitiesKHR& surfaceCapabilities) noexcept
	{
		if (surfaceCapabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) return surfaceCapabilities.currentExtent;
		else
		{
			auto width = int{ 0 };
			auto height = int{ 0 };
			glfwGetFramebufferSize(window, &width, &height);
			const auto suitableWidth = std::clamp(static_cast<uint32_t>(width), surfaceCapabilities.minImageExtent.width, surfaceCapabilities.maxImageExtent.width);
			const auto suitableHeight = std::clamp(static_cast<uint32_t>(height), surfaceCapabilities.minImageExtent.height, surfaceCapabilities.maxImageExtent.height);
			const auto suitableExtent = vk::Extent2D{ suitableWidth, suitableHeight };
			return suitableExtent;
		}
	}
	[[nodiscard]] auto getSuitableQueueFamilies(const vk::PhysicalDevice& physicalDevice, const vk::SurfaceKHR& surface) noexcept
	{
		const auto queueFamiliesProperties = physicalDevice.getQueueFamilyProperties();
		const auto queueFamiliesIndex = std::views::iota(0U, queueFamiliesProperties.size());
		const auto isSuitable = [&](QueueFamilyIndex i)
		{
			const auto isGraphical = physicalDevice.getQueueFamilyProperties()[i].queueFlags & vk::QueueFlagBits::eGraphics;
			const auto isCompute = physicalDevice.getQueueFamilyProperties()[i].queueFlags & vk::QueueFlagBits::eCompute;
			const auto isSurfaceSupported = physicalDevice.getSurfaceSupportKHR(i, surface); // For presentation support
			return isGraphical && isCompute && isSurfaceSupported;
		};
		const auto toQueueFamily = [](QueueFamilyIndex i)
		{
			return QueueFamily{i, {1.0f}}; // One queue of priority level 1
		};
		auto queueFamilies = queueFamiliesIndex | std::views::filter(isSuitable) | std::views::transform(toQueueFamily) | std::ranges::to<std::vector>();
		return queueFamilies;
	}
}

VulkanApplication::VulkanApplication()
	: window{nullptr}
	, instance{}
	, debugMessengerCreateInfo{
		{}
		, vk::DebugUtilsMessageSeverityFlagBitsEXT::eError
			| vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose
			| vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning
		, vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral
			| vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance
			| vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation
		, debugCallback }
	, debugMessenger{}
	, surface{}
	, physicalDevice{}
	, device{}
	, queue{}
	, surfaceFormat{}
	, surfaceExtent{}
	, swapchain{}
	, commandPool{}
	, commandBuffers{}
	, isAcquiredImageReadSemaphore{}
	, isImageRenderedSemaphore{}
	, isCommandBufferExecutedFence{}
{
}

VulkanApplication::~VulkanApplication() noexcept {}

void VulkanApplication::run(const RunInfo& runInfo) noexcept // All exceptions are handled in this function so we can clean up the resources thereafter.
{
	try
	{
		initWindow(); // Must be before initVulkan()
		initVulkan(runInfo);
		// Application info only valid after initVulkan
		const auto applicationInfo = ApplicationInfo{
			surface
			, physicalDevice
			, device
			, queue
			, getSuitableQueueFamilies(physicalDevice, surface)
			, swapchain
			, surfaceFormat.format
			, surfaceExtent
			, commandBuffers
			, isAcquiredImageReadSemaphore
			, isImageRenderedSemaphore
			, isCommandBufferExecutedFence
		};
		runInfo.preRenderLoop(applicationInfo);
		renderLoop(runInfo.renderFrame, applicationInfo);
		runInfo.postRenderLoop(applicationInfo);
		cleanUp(); // Can't put in the class' destructor due to potential exceptions
	}
	catch (const std::exception& except)
	{
		std::cerr << tag::exception << except.what() << std::endl;
	}
}

void VulkanApplication::initWindow()
{
	if (glfwInit() != GLFW_TRUE) throw std::runtime_error{ "Failed to initalize GLFW" };
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // Do not create an OpenGL context
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
	window = glfwCreateWindow(WIDTH, HEIGHT, "MyWindow", nullptr, nullptr);
	if (!window) throw std::runtime_error{ "Can't create window" };
}
void VulkanApplication::initVulkan(const RunInfo& runInfo)
{
	if (glfwVulkanSupported() != GLFW_TRUE) throw std::runtime_error{"Vulkan is not supported"};
	initDispatcher();
	initInstance();
	VULKAN_HPP_DEFAULT_DISPATCHER.init(instance); // Extend dispatcher to support instance dependent EXT function pointers
	initDebugMessenger();
	initSurface();
	initPhysicalDevice();
	initDevice();
	VULKAN_HPP_DEFAULT_DISPATCHER.init(device); // Extend dispatcher to device dependent EXT function pointers
	initQueue();
	initSwapChain(runInfo.swapchainImageUsage);
	initCommandPool();
	initCommandBuffer();
	initSyncObjects();
}
void VulkanApplication::initDispatcher()
{
	const auto dynamicLoader = vk::DynamicLoader{};
	PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = dynamicLoader.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr); // Support instance independent EXT function pointers
}
void VulkanApplication::initInstance()
{
	const auto enabledLayersName = getEnabledLayersName();
	const auto enabledLayersNameProxy = getNamesProxy(enabledLayersName);
	const auto enabledInstanceExtensionsName = getEnabledInstanceExtensionsName();
	const auto enabledInstanceExtensionsNameProxy = getNamesProxy(enabledInstanceExtensionsName);
	const auto applicationInfo = vk::ApplicationInfo {
		"Hello Triangle",
		VK_MAKE_VERSION(1, 0, 0),
		"No Engine",
		VK_MAKE_VERSION(1, 0, 0),
		VK_API_VERSION_1_0
	};
	const auto instanceCreateInfo = vk::InstanceCreateInfo {
		{},
		&applicationInfo,
		enabledLayersNameProxy,
		enabledInstanceExtensionsNameProxy,
		isValidationLayersEnabled ? &debugMessengerCreateInfo : nullptr // Extend create info with debug messenger
	};
	instance = vk::createInstance(instanceCreateInfo);
}
void VulkanApplication::initDebugMessenger()
{
	if (!isValidationLayersEnabled) return;
	debugMessenger = instance.createDebugUtilsMessengerEXT(debugMessengerCreateInfo);
}
void VulkanApplication::initSurface()
{
	auto surfaceProxy = VkSurfaceKHR{};
	const auto result = glfwCreateWindowSurface(instance, window, nullptr, &surfaceProxy);
	if (result != VK_SUCCESS) throw std::runtime_error{ "Failed to create window surface" };
	surface = surfaceProxy;
}
void VulkanApplication::initPhysicalDevice()
{
	const auto physicalDevices = instance.enumeratePhysicalDevices();
	if (physicalDevices.empty()) throw std::runtime_error{"No physical device can be found"};
	physicalDevice = getSuitablePhysicalDevice(physicalDevices, surface);
}
void VulkanApplication::initDevice()
{
	const auto enabledLayersName = getEnabledLayersName();
	const auto enabledLayersNameProxy = getNamesProxy(enabledLayersName);
	const auto enabledDeviceExtensionsName = getDeviceExtensionsName();
	const auto enabledDeviceExtensionsNameProxy = getNamesProxy(enabledDeviceExtensionsName);
	const auto physicalDeviceFeatures = vk::PhysicalDeviceFeatures{};
	const auto queueFamilies = getSuitableQueueFamilies(physicalDevice, surface);
	const auto toQueueCreateInfo = [](const auto& queueFamily)
	{
		const auto& [queueFamilyIndex, queuePriorities] = queueFamily;
		return vk::DeviceQueueCreateInfo{ {}, queueFamilyIndex, queuePriorities };
	};
	const auto queueCreateInfos = queueFamilies | std::views::transform(toQueueCreateInfo) | std::ranges::to<std::vector>();
	const auto deviceCreateInfo = vk::DeviceCreateInfo{
		{},
		queueCreateInfos,
		enabledLayersNameProxy,
		enabledDeviceExtensionsNameProxy,
		&physicalDeviceFeatures
	};
	device = physicalDevice.createDevice(deviceCreateInfo);
}
void VulkanApplication::initQueue()
{
	const auto queueFamilyIndex = 0; // TODO: Always pick the first queueFamilyIndex for now
	const auto queueIndex = 0; // TODO: Always pick the first queue of type queueFamilyIndex for now
	queue = device.getQueue(queueFamilyIndex, queueIndex);
}
void VulkanApplication::initSwapChain(vk::ImageUsageFlagBits swapchainImageUsage)
{
	const auto surfaceAttributes = getSurfaceAttributes(surface, physicalDevice);
	surfaceFormat = getSuitableSurfaceFormat(surfaceAttributes.formats);
	surfaceExtent = getSuitableSurfaceExtent(window, surfaceAttributes.capabilities);
	const auto presentMode = getSuitablePresentMode(surfaceAttributes.presentModes);
	auto imageCount = surfaceAttributes.capabilities.minImageCount + 1;
	if (surfaceAttributes.capabilities.maxImageCount != 0) // not infinite
	{
		imageCount = std::min(imageCount, surfaceAttributes.capabilities.maxImageCount);
	}
	const auto queueFamilies = getSuitableQueueFamilies(physicalDevice, surface);
	vk::SwapchainCreateInfoKHR createInfo{
		{}
		, surface
		, imageCount
		, surfaceFormat.format
		, surfaceFormat.colorSpace
		, surfaceExtent
		, 1 // Single layer, no stereo-rendering
		//, useGraphicPipeline ? vk::ImageUsageFlagBits::eColorAttachment : vk::ImageUsageFlagBits::eTransferDst // Applications that don't use graphic pipeline will most likely use a compute shader to transfer rendered image to the swapchain images
		, swapchainImageUsage
		, vk::SharingMode::eExclusive // Drawing and presentation are done with one physical device's queue
		, queueFamilies.front().first
		, surfaceAttributes.capabilities.currentTransform
		, vk::CompositeAlphaFlagBitsKHR::eOpaque
		, presentMode
		, VK_TRUE
		, VK_NULL_HANDLE
	};
	swapchain = device.createSwapchainKHR(createInfo);
}
void VulkanApplication::initCommandPool()
{
	const auto queueFamilies = getSuitableQueueFamilies(physicalDevice, surface);
	const auto commandPoolCreateInfo = vk::CommandPoolCreateInfo{
		vk::CommandPoolCreateFlagBits::eResetCommandBuffer // reset and rerecord command buffer
		, queueFamilies.front().first
	};
	commandPool = device.createCommandPool(commandPoolCreateInfo);
}
void VulkanApplication::initCommandBuffer()
{
	const auto allocateInfo = vk::CommandBufferAllocateInfo{
		commandPool
		, vk::CommandBufferLevel::ePrimary
		, 1
	};
	// Command buffers are allocated from the command pool
	commandBuffers = device.allocateCommandBuffers(allocateInfo);
}
void VulkanApplication::initSyncObjects()
{
	const auto semCreateInfo = vk::SemaphoreCreateInfo{};
	const auto fenceCreateInfo = vk::FenceCreateInfo{vk::FenceCreateFlagBits::eSignaled}; // First frame doesn't have to wait for the unexisted previous image
	isAcquiredImageReadSemaphore = device.createSemaphore(semCreateInfo);
	isImageRenderedSemaphore = device.createSemaphore(semCreateInfo);
	isCommandBufferExecutedFence = device.createFence(fenceCreateInfo);
}

void VulkanApplication::renderLoop(const ApplicationFunction& renderFrame, const ApplicationInfo& applicationInfo)
{
	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();
		renderFrame(applicationInfo);
	}
	device.waitIdle(); // Wait for the queue(s) to become idle, i.e. finished executing the cmds?
}

void VulkanApplication::cleanUp() // Destroy the objects in reverse order of their creation order
{
	device.destroy(isAcquiredImageReadSemaphore);
	device.destroy(isImageRenderedSemaphore);
	device.destroy(isCommandBufferExecutedFence);
	device.destroyCommandPool(commandPool);
	device.destroySwapchainKHR(swapchain);
	device.destroy();
	if (isValidationLayersEnabled) instance.destroyDebugUtilsMessengerEXT(debugMessenger);
	instance.destroySurfaceKHR(surface);
	instance.destroy();
	glfwDestroyWindow(window);
	glfwTerminate();
}


