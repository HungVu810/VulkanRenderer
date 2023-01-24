#include "VulkanApplication.h"

VulkanApplication::VulkanApplication()
	: window{ nullptr }
	, instance{}
	, applicationInfo{"Hello Triangle", VK_MAKE_VERSION(1, 0, 0), "No Engine", VK_MAKE_VERSION(1, 0, 0), VK_API_VERSION_1_0 }
	, requiredExtensionNames{} // depended on glfw
	, requiredLayerNames{} // depended on enableValidationLayers
	, instanceCreateInfo{}
{
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
	cleanup();
}

auto VulkanApplication::initWindow() -> void
{
	assertm(glfwInit() == GLFW_TRUE, "Failed to initalize GLFW");
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // Do not create an OpenGL context
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
	window = glfwCreateWindow(WIDTH, HEIGHT, "MyWindow", nullptr, nullptr);
	assertm(window, "Can't create window");
}

auto VulkanApplication::isValidationLayerSupported() const noexcept -> bool
{
	std::vector<vk::LayerProperties> availableLayers = vk::enumerateInstanceLayerProperties();
	const auto extractLayerName = [](const VkLayerProperties& availLayer) { return std::string_view(availLayer.layerName); };
	const auto availableLayersName = availableLayers | std::views::transform(extractLayerName);
	const auto isFound = [&](std::string_view validationLayerName)
	{
		return std::ranges::find(availableLayersName, validationLayerName) != std::end(availableLayersName);
	};
	return std::ranges::all_of(requiredLayerNames, isFound);
}
auto VulkanApplication::setRequiredLayerNames() noexcept -> void
{
	if (enableValidationLayers)
	{
		requiredLayerNames.push_back("VK_LAYER_KHRONOS_validation");
		assertm(isValidationLayerSupported(), "Validation layers requested, but not available!");
	}
}
auto VulkanApplication::setRequiredExtensionNames() -> std::vector<const char*> 
{
	uint32_t extensionCount = 0;
	const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&extensionCount);
	assertm(glfwExtensions, "Vulkan's surface extensions for window creation can not be found");
	requiredExtensionNames = std::vector<const char*>(glfwExtensions, glfwExtensions + extensionCount);
	if (enableValidationLayers)
	{
		requiredExtensionNames.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
	}
	return requiredExtensionNames;
}
auto VulkanApplication::setInstanceCreateInfo() -> void
{
	assertm(glfwVulkanSupported() == GLFW_TRUE, "Vulkan is not supported");
	setRequiredExtensionNames();
	setRequiredLayerNames();
	if (enableValidationLayers)
	{
		assertm(isValidationLayerSupported(), "Validation layers requested, but not available!");
		instanceCreateInfo.setPEnabledLayerNames(requiredLayerNames);
	}
	else instanceCreateInfo.enabledLayerCount = 0;
	instanceCreateInfo.pApplicationInfo = &applicationInfo;
	instanceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(requiredExtensionNames.size());
	instanceCreateInfo.ppEnabledExtensionNames = requiredExtensionNames.data();
}
auto VulkanApplication::createVulkanInstance() -> void
{
	setInstanceCreateInfo();
	assertm(vk::createInstance(&instanceCreateInfo, nullptr, &instance) == vk::Result::eSuccess, "failed to create instance!");
	//TODO: Debugging instance creation and destruction
}
auto VulkanApplication::initVulkan() -> void
{
	createVulkanInstance();
	//setupDebugMessenger();
}

auto VulkanApplication::mainLoop() -> void
{
	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();
	}
}

auto VulkanApplication::cleanup() -> void
{
	vkDestroyInstance(instance, nullptr);
	glfwDestroyWindow(window);
	glfwTerminate();
}



