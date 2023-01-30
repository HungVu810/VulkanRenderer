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
	// instanceCreateInfo args
	, extension{} // depended on glfw
	, layer{} // depended on enableValidationLayers
	, debugMessengerCreateInfo{
		{}
		, vk::DebugUtilsMessageSeverityFlagBitsEXT::eError
			| vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose
			| vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning
		, vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral
			| vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance
			| vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation
		, debugCallback }
	// instanceCreateInfo args
	, instanceCreateInfo{}
	, instance{}
	, debugMessenger{}
	, queueCreateInfos{}
	, device{}
{
}
VulkanApplication::~VulkanApplication()
{
	if (enableValidationLayers)
	{
		instance.destroyDebugUtilsMessengerEXT(debugMessenger);
	}
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
auto VulkanApplication::initLayerNames() noexcept -> void
{
	if (!enableValidationLayers) return;
	std::string validationLayerName = "VK_LAYER_KHRONOS_validation";
	std::vector<vk::LayerProperties> availableLayers = vk::enumerateInstanceLayerProperties();
	assertm(!availableLayers.empty(), "No layer can be found");
	const auto extractLayerName = [](const vk::LayerProperties& availLayer) { return std::string_view(availLayer.layerName); };
	const auto availableLayersName = availableLayers | std::views::transform(extractLayerName);
	bool isFound = std::ranges::find(availableLayersName, validationLayerName) != std::end(availableLayersName);
	assertm(isFound, "Validation layers requested, but is not supported!");
	layer.names.push_back(validationLayerName);
}
auto VulkanApplication::initExtensionNames() noexcept -> void
{
	uint32_t extensionCount = 0;
	const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&extensionCount);
	assertm(glfwExtensions, "Vulkan's surface extensions for window creation can not be found");
	extension.names = std::vector<std::string>(glfwExtensions, glfwExtensions + extensionCount);
	if (enableValidationLayers)
	{
		extension.names.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
	}
}
auto VulkanApplication::initInstanceCreateInfo() -> void {
	assertm(glfwVulkanSupported() == GLFW_TRUE, "Vulkan is not supported");
	initLayerNames();
	initExtensionNames();
	instanceCreateInfo.setPApplicationInfo(&applicationInfo);
	instanceCreateInfo.setPEnabledLayerNames(layer.getProxy());
	instanceCreateInfo.setPEnabledExtensionNames(extension.getProxy());
	if (!enableValidationLayers) return;
	instanceCreateInfo.setPNext(&debugMessengerCreateInfo); // extend instance create info with debug messenger
}
auto VulkanApplication::getSuitablePhysicalDevice() const -> vk::PhysicalDevice
{
	std::vector<vk::PhysicalDevice> physicalDevices = instance.enumeratePhysicalDevices();
	assertm(!physicalDevices.empty(), "No physical device can be found");
	const auto isSuitable = [](const vk::PhysicalDevice& physicalDevice)
	{
		vk::PhysicalDeviceProperties properties{ physicalDevice.getProperties() };
		vk::PhysicalDeviceFeatures features{ physicalDevice.getFeatures() };
		return properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu
			&& features.geometryShader;
	};
	const auto physicalDeviceIter = std::ranges::find_if(physicalDevices, isSuitable);
	assertm(physicalDeviceIter != physicalDevices.end(), "Can't find a suitable GPU");
	return *physicalDeviceIter;
}
auto VulkanApplication::initQueueCreateInfos(const vk::PhysicalDevice& physicalDevice) -> void
{
	const auto queueFamilies = physicalDevice.getQueueFamilyProperties();
	std::vector<uint32_t> suitableQueueFamiles;
	for (uint32_t i = 0; i < queueFamilies.size(); i++)
	{
		if (queueFamilies[i].queueFlags & vk::QueueFlagBits::eGraphics)
		{
			suitableQueueFamiles.push_back(i);
		}
	}
	assertm(!suitableQueueFamiles.empty(), "No desired queue family found");
	queuePriority = { 1.0f };
	const vk::DeviceQueueCreateInfo queueCreateInfo{ {}, suitableQueueFamiles.front(), queuePriority };
	queueCreateInfos.push_back(queueCreateInfo);
}
auto VulkanApplication::initDevice() -> void
{
	const vk::PhysicalDevice physicalDevice{ getSuitablePhysicalDevice() };
	const vk::PhysicalDeviceFeatures physicalDeviceFeatures{};
	initQueueCreateInfos(physicalDevice);
	vk::DeviceCreateInfo deviceCreateInfo{ {}, queueCreateInfos, layer.getProxy(), nullptr, &physicalDeviceFeatures };
	device = physicalDevice.createDevice(deviceCreateInfo);
}
auto VulkanApplication::initDebugMessenger() -> void
{
	if (!enableValidationLayers) return;
	vk::Result result = instance.createDebugUtilsMessengerEXT(&debugMessengerCreateInfo, nullptr, &debugMessenger);
	assertm(result == vk::Result::eSuccess, "Failed to create debug messenger");
}
auto VulkanApplication::initVulkan() -> void
{
	initDispatcher();
	initInstanceCreateInfo();
	instance = vk::createInstance(instanceCreateInfo);
	VULKAN_HPP_DEFAULT_DISPATCHER.init(instance); // extend dispatcher to support instance dependent EXT function pointers
	initDevice();
	VULKAN_HPP_DEFAULT_DISPATCHER.init(device); // extend dispatcher to device dependent EXT function pointers
	initDebugMessenger();
}

auto VulkanApplication::mainLoop() -> void
{
	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();
	}
}




