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
{
}
VulkanApplication::~VulkanApplication()
{
	if (enableValidationLayers)
	{
		instance.destroyDebugUtilsMessengerEXT(debugMessenger);
	}
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
	const auto extractLayerName = [](const VkLayerProperties& availLayer) { return std::string_view(availLayer.layerName); };
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
}
auto VulkanApplication::extendInstanceCreateInfo() -> void
{
	if (!enableValidationLayers) return;
	instanceCreateInfo.setPNext(&debugMessengerCreateInfo);
}
auto VulkanApplication::initInstance() -> void
{
	vk::Result result = vk::createInstance(&instanceCreateInfo, nullptr, &instance);
	assertm(result == vk::Result::eSuccess, "failed to create instance!");
}
auto VulkanApplication::extendDispatcher() -> void
{
	VULKAN_HPP_DEFAULT_DISPATCHER.init(instance); // support instance dependent EXT function pointers
	//std::vector<vk::PhysicalDevice> physicalDevices = instance.enumeratePhysicalDevices();
	//assert(!physicalDevices.empty());
	//vk::Device device = physicalDevices[0].createDevice({}, nullptr);
	//VULKAN_HPP_DEFAULT_DISPATCHER.init(device); // device dependent EXT function pointers
}
auto VulkanApplication::initDebugMessenger() -> void
{
	if (!enableValidationLayers) return;
	vk::Result result = instance.createDebugUtilsMessengerEXT(&debugMessengerCreateInfo, nullptr, &debugMessenger);
	assertm(result == vk::Result::eSuccess, "failed to create debug messenger");
}
auto VulkanApplication::initVulkan() -> void
{
	initDispatcher();
	initInstanceCreateInfo();
	extendInstanceCreateInfo();
	initInstance();
	extendDispatcher();
	initDebugMessenger();

}

auto VulkanApplication::mainLoop() -> void
{
	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();
	}
}




