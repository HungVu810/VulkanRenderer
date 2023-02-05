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
	, device{}
	, queue{}
	, surface{}
{
}
VulkanApplication::~VulkanApplication()
{
	if (enableValidationLayers)
	{
		instance.destroyDebugUtilsMessengerEXT(debugMessenger);
	}
	instance.destroySurfaceKHR(surface);
	device.destroy();
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
// initInstanceCreateInfo --------- START
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
auto VulkanApplication::initInstanceCreateInfo() -> void
{
	initLayer();
	initInstanceExtension();
	instanceCreateInfo.setPApplicationInfo(&applicationInfo);
	instanceCreateInfo.setPEnabledLayerNames(layer.getProxy());
	instanceCreateInfo.setPEnabledExtensionNames(instanceExtension.getProxy());
	if (!enableValidationLayers) return;
	instanceCreateInfo.setPNext(&debugMessengerCreateInfo); // extend instance create info with debug messenger
}
// initInstanceCreateInfo --------- END
auto VulkanApplication::initDebugMessenger() -> void
{
	if (!enableValidationLayers) return;
	vk::Result result = instance.createDebugUtilsMessengerEXT(&debugMessengerCreateInfo, nullptr, &debugMessenger);
	assertm(result == vk::Result::eSuccess, "Failed to create debug messenger");
}
auto VulkanApplication::initWindowSurface() -> void
{
	VkSurfaceKHR surfaceTemp;
	VkResult result = glfwCreateWindowSurface(instance, window, nullptr, &surfaceTemp);
	assertm(result == VK_SUCCESS, "Failed to create window surface");
	surface = surfaceTemp;
}
// initDevice --------- START
auto VulkanApplication::initDeviceExtension() noexcept -> void
{
	deviceExtension.names.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
}
auto VulkanApplication::getSuitablePhysicalDevice() const -> vk::PhysicalDevice
{
	std::vector<vk::PhysicalDevice> physicalDevices = instance.enumeratePhysicalDevices();
	assertm(!physicalDevices.empty(), "No physical device can be found");
	const auto isSuitable = [this](const vk::PhysicalDevice& physicalDevice)
	{
		vk::PhysicalDeviceProperties properties{ physicalDevice.getProperties() };
		vk::PhysicalDeviceFeatures features{ physicalDevice.getFeatures() };
		const auto extractExtensionsName = [](const vk::ExtensionProperties& properties) { return static_cast<std::string_view>(properties.extensionName); };
		const auto physicalDeviceExtensionsName = physicalDevice.enumerateDeviceExtensionProperties() | std::views::transform(extractExtensionsName);
		const auto isSupported = [&physicalDeviceExtensionsName](const auto& requiredExtension) { return std::ranges::find(physicalDeviceExtensionsName, requiredExtension) != physicalDeviceExtensionsName.end(); };
		return properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu
			&& features.geometryShader
			&& std::ranges::all_of(deviceExtension.names, isSupported);
	};
	const auto physicalDeviceIter = std::ranges::find_if(physicalDevices, isSuitable);
	assertm(physicalDeviceIter != physicalDevices.end(), "Can't find a suitable GPU");
	return *physicalDeviceIter;
}
auto VulkanApplication::initQueueCreateInfos(const vk::PhysicalDevice& physicalDevice) noexcept -> void
{
	const auto physicalDeviceQueueFamilies = physicalDevice.getQueueFamilyProperties();
	const auto queueFamiliesIndex = std::views::iota(0u, physicalDeviceQueueFamilies.size());
	const auto findSuitableIndex = [&](uint32_t i)
	{
		const auto isGraphical = physicalDeviceQueueFamilies[i].queueFlags & vk::QueueFlagBits::eGraphics;
		const auto isSurfaceSupported = physicalDevice.getSurfaceSupportKHR(i, surface);
		return isGraphical && isSurfaceSupported;
	};
	const auto populateQueueFamilies = [this](uint32_t i)
	{
		queueFamilies.push_back({ i, { 1.0f } });
	};
	std::ranges::for_each(queueFamiliesIndex | std::views::filter(findSuitableIndex), populateQueueFamilies);
	assertm(!queueFamilies.empty(), "No desired queue family index found");
	const auto populateQueueCreateInfos = [this](const auto& queueFamily)
	{
		const auto& [familyIndex, queues] = queueFamily;
		const vk::DeviceQueueCreateInfo queueCreateInfo{ {}, familyIndex, queues };
		queueCreateInfos.push_back(queueCreateInfo);
	};
	std::ranges::for_each(queueFamilies, populateQueueCreateInfos);
}
auto VulkanApplication::initDevice() -> void
{
	initDeviceExtension();
	const vk::PhysicalDevice physicalDevice{ getSuitablePhysicalDevice() };
	const vk::PhysicalDeviceFeatures physicalDeviceFeatures{};
	initQueueCreateInfos(physicalDevice);
	vk::DeviceCreateInfo deviceCreateInfo{ {}, queueCreateInfos, layer.getProxy(), deviceExtension.getProxy(), &physicalDeviceFeatures};
	device = physicalDevice.createDevice(deviceCreateInfo);
}
// initDevice --------- END
auto VulkanApplication::initQueue() -> void
{
	int queueFamilyIndex = 0; // TODO: always pick the first one tentatively for now
	int queueIndex = 0; // TODO: always pick the first one tentatively for now
	queue = device.getQueue(queueFamilyIndex, queueIndex);
}
auto VulkanApplication::initVulkan() -> void
{
	assertm(glfwVulkanSupported() == GLFW_TRUE, "Vulkan is not supported");
	initDispatcher();
	initInstanceCreateInfo();
	instance = vk::createInstance(instanceCreateInfo);
	VULKAN_HPP_DEFAULT_DISPATCHER.init(instance); // extend dispatcher to support instance dependent EXT function pointers
	initDebugMessenger();
	initWindowSurface();
	initDevice();
	VULKAN_HPP_DEFAULT_DISPATCHER.init(device); // extend dispatcher to device dependent EXT function pointers
	initQueue();
}

auto VulkanApplication::mainLoop() -> void
{
	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();
	}
}




