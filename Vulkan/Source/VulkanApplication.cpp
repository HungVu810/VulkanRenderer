#include "VulkanApplication.h"
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

VulkanApplication::VulkanApplication()
	: window{ nullptr }
	, applicationInfo{
		"Hello Triangle"
		, VK_MAKE_VERSION(1, 0, 0)
		, "No Engine"
		, VK_MAKE_VERSION(1, 0, 0)
		, VK_API_VERSION_1_0 }
	, layer{}, instanceExtension{}, deviceExtension{}
	, debugMessengerCreateInfo{
		{}
		, vk::DebugUtilsMessageSeverityFlagBitsEXT::eError
			| vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose
			| vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning
		, vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral
			| vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance
			| vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation
		, debugCallback }
	, instanceCreateInfo{}
	, instance{}
	, debugMessenger{}
	, queueFamilies{}
	, queueCreateInfos{}
	, physicalDevice{}
	, device{}
	, queue{}
	, surface{}
	, swapChain{}
{
}
VulkanApplication::~VulkanApplication()
{
	device.destroySwapchainKHR(swapChain);
	device.destroy();
	if (enableValidationLayers)
	{
		instance.destroyDebugUtilsMessengerEXT(debugMessenger);
	}
	instance.destroySurfaceKHR(surface);
	instance.destroy();
	glfwDestroyWindow(window);
	glfwTerminate();
}

auto VulkanApplication::run() noexcept -> void
{
	try
	{
		initWindow(); // must be before initVulkan()
		initVulkan();
		mainLoop();
	}
	catch (const std::exception& except)
	{
		std::cerr << except.what() << std::endl;
	}
}

auto VulkanApplication::initWindow() -> void
{
	assertm(glfwInit() == GLFW_TRUE, "Failed to initalize GLFW");
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // Do not create an OpenGL context
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
	window = glfwCreateWindow(WIDTH, HEIGHT, "MyWindow", nullptr, nullptr);
	assertm(window, "Can't create window");
}

auto VulkanApplication::initDispatcher() -> void
{
	vk::DynamicLoader dynamicLoader;
	PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = dynamicLoader.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr); // support instance independent EXT function pointers
}
// initInstance --------- START
auto VulkanApplication::initLayer() noexcept -> void
{
	if (!enableValidationLayers) return;
	std::string validationLayerName = "VK_LAYER_KHRONOS_validation";
	std::vector<vk::LayerProperties> availableLayers = vk::enumerateInstanceLayerProperties();
	assertm(!availableLayers.empty(), "No layer can be found");
	const auto extractLayerName = [](const vk::LayerProperties& availLayer) { return std::string_view(availLayer.layerName); };
	const auto availableLayersName = availableLayers | std::views::transform(extractLayerName);
	bool isSupported = std::ranges::find(availableLayersName, validationLayerName) != std::end(availableLayersName);
	assertm(isSupported, "Validation layers requested, but is not supported!");
	layer.names.push_back(validationLayerName);
}
auto VulkanApplication::initInstanceExtension() noexcept -> void
{
	uint32_t extensionCount = 0;
	const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&extensionCount);
	assertm(glfwExtensions, "Vulkan's surface extensions for window creation can not be found");
	instanceExtension = std::vector<std::string>(glfwExtensions, glfwExtensions + extensionCount);
	if (enableValidationLayers)
	{
		instanceExtension.names.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
	}
}
auto VulkanApplication::initInstance() -> void
{
	initLayer();
	initInstanceExtension();
	instanceCreateInfo.setPApplicationInfo(&applicationInfo);
	instanceCreateInfo.setPEnabledLayerNames(layer.getProxy());
	instanceCreateInfo.setPEnabledExtensionNames(instanceExtension.getProxy());
	if (!enableValidationLayers) return;
	instanceCreateInfo.setPNext(&debugMessengerCreateInfo); // extend instance create info with debug messenger
	instance = vk::createInstance(instanceCreateInfo);
}
// initInstance --------- END
auto VulkanApplication::initDebugMessenger() -> void
{
	if (!enableValidationLayers) return;
	vk::Result result = instance.createDebugUtilsMessengerEXT(&debugMessengerCreateInfo, nullptr, &debugMessenger);
	assertm(result == vk::Result::eSuccess, "Failed to create debug messenger");
}
auto VulkanApplication::initSurface() -> void
{
	VkSurfaceKHR surfaceTemp;
	VkResult result = glfwCreateWindowSurface(instance, window, nullptr, &surfaceTemp);
	assertm(result == VK_SUCCESS, "Failed to create window surface");
	surface = surfaceTemp;
}
auto getSurfaceAttributes(const vk::SurfaceKHR& surface, const vk::PhysicalDevice& physicalDevice) noexcept -> SurfaceAttributes
{
	return SurfaceAttributes{
		.capabilities = physicalDevice.getSurfaceCapabilitiesKHR(surface),
		.formats      = physicalDevice.getSurfaceFormatsKHR(surface),
		.presentModes = physicalDevice.getSurfacePresentModesKHR(surface)
	};
}
// initPhysicalDevice --------- START
auto VulkanApplication::initDeviceExtension() noexcept -> void
{
	deviceExtension.names.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
}
[[nodiscard]] auto getSuitablePhysicalDevice(const vk::Instance& instance, const std::function<bool (const vk::PhysicalDevice&)>& isSuitable) -> vk::PhysicalDevice
{
	std::vector<vk::PhysicalDevice> physicalDevices = instance.enumeratePhysicalDevices();
	assertm(!physicalDevices.empty(), "No physical device can be found");
	const auto physicalDeviceIter = std::ranges::find_if(physicalDevices, isSuitable);
	assertm(physicalDeviceIter != physicalDevices.end(), "Can't find a suitable GPU");
	return *physicalDeviceIter;
}
auto VulkanApplication::initQueueCreateInfos(const vk::PhysicalDevice& physicalDevice) noexcept -> void
{
	const auto physicalDeviceQueueFamilies = physicalDevice.getQueueFamilyProperties();
	const auto queueFamiliesIndex = std::views::iota(0u, physicalDeviceQueueFamilies.size());
	const auto suitableIndex = [&](uint32_t i)
	{
		const auto isGraphical = physicalDeviceQueueFamilies[i].queueFlags & vk::QueueFlagBits::eGraphics;
		const auto isSurfaceSupported = physicalDevice.getSurfaceSupportKHR(i, surface); // for presentation
		return isGraphical && isSurfaceSupported;
	};
	const auto toQueueFamily = [](uint32_t i) { return QueueFamily{ i, { 1.0f } }; };
	std::ranges::move(
		queueFamiliesIndex
		| std::views::filter(suitableIndex)
		| std::views::transform(toQueueFamily)
		, std::back_inserter(queueFamilies)
	);
	assertm(!queueFamilies.empty(), "No desired queue family index found");
	const auto toDeviceQueueCreateInfo = [](const auto& queueFamily) {
		const auto& [familyIndex, queues] = queueFamily;
		return vk::DeviceQueueCreateInfo{ {}, familyIndex, queues };
	};
	std::ranges::move(
		queueFamilies
		| std::views::transform(toDeviceQueueCreateInfo)
		, std::back_inserter(queueCreateInfos)
	);
}
auto VulkanApplication::initPhysicalDevice() -> void
{
	initDeviceExtension();
	const auto isSuitable = [this](const vk::PhysicalDevice& physicalDevice)
	{
		vk::PhysicalDeviceProperties properties{ physicalDevice.getProperties() };
		vk::PhysicalDeviceFeatures features{ physicalDevice.getFeatures() };
		const auto extractExtensionsName = [](const vk::ExtensionProperties& properties) { return static_cast<std::string_view>(properties.extensionName); };
		const auto physicalDeviceExtensionsName = physicalDevice.enumerateDeviceExtensionProperties() | std::views::transform(extractExtensionsName);
		const auto isSupported = [&physicalDeviceExtensionsName](const auto& requiredExtensionName) { return std::ranges::find(physicalDeviceExtensionsName, requiredExtensionName) != physicalDeviceExtensionsName.end(); };
		const auto surfaceAttributes{ getSurfaceAttributes(surface, physicalDevice) };
		return properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu
			&& features.geometryShader
			&& std::ranges::all_of(deviceExtension.names, isSupported)
			&& !surfaceAttributes.formats.empty() // --------- TODO: NOW CHECK FOR THE FORMATS
			&& !surfaceAttributes.presentModes.empty(); // --- TODO: NOW CHECK FOR THE PRESENTATION MODES
	};
	physicalDevice = getSuitablePhysicalDevice(instance, isSuitable);
}
// initPhysicalDevice --------- END
auto VulkanApplication::initDevice() -> void
{
	const vk::PhysicalDeviceFeatures physicalDeviceFeatures{};
	initQueueCreateInfos(physicalDevice);
	vk::DeviceCreateInfo deviceCreateInfo{ {}, queueCreateInfos, layer.getProxy(), deviceExtension.getProxy(), &physicalDeviceFeatures};
	device = physicalDevice.createDevice(deviceCreateInfo);
}
auto VulkanApplication::initQueue() -> void
{
	int QueueFamilyIndex = 0; // TODO: always pick the first one tentatively for now
	int queueIndex = 0; // TODO: always pick the first one tentatively for now
	queue = device.getQueue(QueueFamilyIndex, queueIndex);
}
// initSwapChain --------- START
[[nodiscard]] auto findSuitableSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& surfaceFormats) -> vk::SurfaceFormatKHR
{
	const auto isSuitable = [](const auto& surfaceFormat)
	{
		return surfaceFormat.format == vk::Format::eB8G8R8A8Srgb
			&& surfaceFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear;
	};
	const auto surfaceFormatsIter = std::ranges::find_if(surfaceFormats, isSuitable);
	if (surfaceFormatsIter == surfaceFormats.end()) return surfaceFormats.at(0);
	else return *surfaceFormatsIter;
}
[[nodiscard]] auto findSuitablePresentMode(const std::vector<vk::PresentModeKHR>& presentModes) -> vk::PresentModeKHR
{
	const auto isSuitable = [](const auto& presentMode)
	{
		return presentMode == vk::PresentModeKHR::eMailbox;
	};
	const auto presentModesIter = std::ranges::find_if(presentModes, isSuitable);
	if (presentModesIter == presentModes.end()) return vk::PresentModeKHR::eFifo;
	else return *presentModesIter;
}
[[nodiscard]] auto findSuitableSurfaceExtent(GLFWwindow* window, const vk::SurfaceCapabilitiesKHR& surfaceCapabilities) -> vk::Extent2D
{
	if (surfaceCapabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
	{
		return surfaceCapabilities.currentExtent;
	}
	else
	{
		int width, height;
		glfwGetFramebufferSize(window, &width, &height);
		VkExtent2D actualExtent = { static_cast<uint32_t>(width), static_cast<uint32_t>(height) };
		actualExtent.width = std::clamp(actualExtent.width, surfaceCapabilities.minImageExtent.width, surfaceCapabilities.maxImageExtent.width);
		actualExtent.height = std::clamp(actualExtent.height, surfaceCapabilities.minImageExtent.height, surfaceCapabilities.maxImageExtent.height);
		return actualExtent;
	}
}
auto VulkanApplication::initSwapChain() -> void
{
	const auto surfaceAttributes = getSurfaceAttributes(surface, physicalDevice);
	const auto surfaceFormat = findSuitableSurfaceFormat(surfaceAttributes.formats);
	const auto presentMode = findSuitablePresentMode(surfaceAttributes.presentModes);
	const auto surfaceExtent = findSuitableSurfaceExtent(window, surfaceAttributes.capabilities);
	auto imageCount = surfaceAttributes.capabilities.minImageCount + 1;
	if (surfaceAttributes.capabilities.maxImageCount != 0) // not infinite
	{
		imageCount = std::min(imageCount, surfaceAttributes.capabilities.maxImageCount);
	}
	vk::SwapchainCreateInfoKHR createInfo{
		{}
		, surface
		, imageCount // number of framebuffers
		, surfaceFormat.format
		, surfaceFormat.colorSpace
		, surfaceExtent // width and height for the framebuffers
		, 1 // specifies the amount of image's layers (ie: more than 1 for 3D stereoscopic application)
		, vk::ImageUsageFlagBits::eColorAttachment // render directly onto the framebuffers (no post-processing)
		, vk::SharingMode::eExclusive // drawing and presentation are done with one physical device's queue
		, queueFamilies.front().first
		, surfaceAttributes.capabilities.currentTransform
		, vk::CompositeAlphaFlagBitsKHR::eOpaque
		, presentMode
		, VK_TRUE
		, VK_NULL_HANDLE
	};
	swapChain = device.createSwapchainKHR(createInfo);
}
// initSwapChain --------- END
auto VulkanApplication::initVulkan() -> void
{
	assertm(glfwVulkanSupported() == GLFW_TRUE, "Vulkan is not supported");
	initDispatcher();
	initInstance();
	VULKAN_HPP_DEFAULT_DISPATCHER.init(instance); // extend dispatcher to support instance dependent EXT function pointers
	initDebugMessenger();
	initSurface();
	initPhysicalDevice();
	initDevice();
	VULKAN_HPP_DEFAULT_DISPATCHER.init(device); // extend dispatcher to device dependent EXT function pointers
	initQueue();
	initSwapChain();
}

auto VulkanApplication::mainLoop() -> void
{
	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();
	}
}




