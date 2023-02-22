#include "VulkanApplication.h"
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace
{
	// Class/Structs
	auto getNamesProxy(const std::vector<std::string>& names) noexcept -> std::vector<const char*>
	{
		std::vector<const char*> proxy;
		const auto toCStr = [](std::string_view str) { return str.data(); };
		std::ranges::move(names | std::views::transform(toCStr), std::back_inserter(proxy));
		return proxy;
	}
	class Names
	{
	public:
		[[nodiscard]] Names() : names{}, proxy{} {};
		[[nodiscard]] Names(const std::vector<std::string>& strings) noexcept : names{ strings }, proxy{ getNamesProxy(names) } {}
		[[nodiscard]] Names(std::vector<std::string>&& strings) noexcept : names{ strings }, proxy{ getNamesProxy(names) } {}
		[[nodiscard]] Names(const Names& inNames) noexcept : names{ inNames.names }, proxy{ getNamesProxy(names) } {}
		[[nodiscard]] Names(const Names&& inNames) noexcept : names{ inNames.names }, proxy{ getNamesProxy(names) } {}
		// Todo: deducing this C++ 23 for both this & const this
		//[[nodiscard]] auto getProxy() noexcept -> const std::vector<const char*>&
		//{
		//	proxy.clear();
		//	populateNamesProxy(names, proxy);
		//	return proxy;
		//}
		// This function is deleted because to use it, we must either return a
		// temporary proxy, which is UB because the local proxy is detroyed after
		// the function returned and its data is moved/copied to variable calling
		// this function, or we must accept an external proxy variable so that its
		// lifetime will out live this function.
		// Consider making Names non-const
		// [[nodiscard]] auto getProxy() const noexcept -> const std::vector<const char*> = delete;
		[[nodiscard]] auto getProxy() const noexcept -> const std::vector<const char*>&
		{
			return proxy;
		}

	public:
		std::vector<std::string> names;

	private:
		std::vector<const char*> proxy; // Used with the Vulkan API, which is just vector names's std::string casted to const char* const
	};
	struct SurfaceAttributes
	{
		vk::SurfaceCapabilitiesKHR capabilities;
		std::vector<vk::SurfaceFormatKHR> formats;
		std::vector<vk::PresentModeKHR> presentModes;
	};

	// Helper functions
	using QueueFamilyIndex = uint32_t;
	using QueuesPriorities = std::vector<float>;
	using QueueFamily = std::pair<QueueFamilyIndex, QueuesPriorities>;
	[[nodiscard]] auto getLayer() noexcept -> Names
	{
		if (!enableValidationLayers) return Names{};
		std::string validationLayerName = "VK_LAYER_KHRONOS_validation";
		std::vector<vk::LayerProperties> layersProperties = vk::enumerateInstanceLayerProperties();
		assertm(!layersProperties.empty(), "No supported layers can be found");
		const auto getLayerName = [](const vk::LayerProperties& availLayer) { return std::string_view(availLayer.layerName); };
		const auto layersName = layersProperties | std::views::transform(getLayerName);
		bool isSupported = std::ranges::find(layersName, validationLayerName) != std::end(layersName);
		assertm(isSupported, "Validation layers requested, but is not supported!");
		std::vector<std::string> names { validationLayerName };
		return Names{ names };
	}
	[[nodiscard]] auto getInstanceExtension() noexcept -> Names
	{
		Names instanceExtension{};
		uint32_t extensionCount = 0;
		const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&extensionCount);
		assertm(glfwExtensions, "Vulkan's surface extensions for window creation can not be found");
		std::vector<std::string> names(glfwExtensions, glfwExtensions + extensionCount);
		if (enableValidationLayers)
		{
			names.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
		}
		return Names{ names };
	}
	[[nodiscard]] auto getDeviceExtension() noexcept -> Names
	{
		return Names{ { VK_KHR_SWAPCHAIN_EXTENSION_NAME } };
	}
	[[nodiscard]] auto getSuitablePhysicalDevice(const vk::Instance& instance, const std::function<bool (const vk::PhysicalDevice&)>& isSuitable) -> vk::PhysicalDevice
	{
		std::vector<vk::PhysicalDevice> physicalDevices = instance.enumeratePhysicalDevices();
		assertm(!physicalDevices.empty(), "No physical device can be found");
		const auto physicalDeviceIter = std::ranges::find_if(physicalDevices, isSuitable);
		assertm(physicalDeviceIter != physicalDevices.end(), "Can't find a suitable GPU");
		return *physicalDeviceIter;
	}
	[[nodiscard]] auto getSuitableSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& surfaceFormats) -> vk::SurfaceFormatKHR
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
	[[nodiscard]] auto getSuitablePresentMode(const std::vector<vk::PresentModeKHR>& presentModes) -> vk::PresentModeKHR
	{
		const auto isSuitable = [](const auto& presentMode)
		{
			return presentMode == vk::PresentModeKHR::eMailbox;
		};
		const auto presentModesIter = std::ranges::find_if(presentModes, isSuitable);
		if (presentModesIter == presentModes.end()) return vk::PresentModeKHR::eFifo;
		else return *presentModesIter;
	}
	[[nodiscard]] auto getSuitableSurfaceExtent(GLFWwindow* window, const vk::SurfaceCapabilitiesKHR& surfaceCapabilities) -> vk::Extent2D
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
	[[nodiscard]] auto getSurfaceAttributes(const vk::SurfaceKHR& surface, const vk::PhysicalDevice& physicalDevice) noexcept -> SurfaceAttributes
	{
		return SurfaceAttributes{
			.capabilities = physicalDevice.getSurfaceCapabilitiesKHR(surface),
			.formats      = physicalDevice.getSurfaceFormatsKHR(surface),
			.presentModes = physicalDevice.getSurfacePresentModesKHR(surface)
		};
	}
	[[nodiscard]] auto getQueueFamilies(const vk::PhysicalDevice& physicalDevice, const std::function<bool(QueueFamilyIndex i)>& isSuitable) noexcept -> std::vector<QueueFamily>
	{
		const auto queueFamiliesProperties = physicalDevice.getQueueFamilyProperties();
		const auto queueFamiliesIndex = std::views::iota(0u, queueFamiliesProperties.size());
		const auto toQueueFamily = [](QueueFamilyIndex i) {
			return QueueFamily{ i, { 1.0f } }; // one queue of priority level 1.0f
		};
		std::vector<QueueFamily> queueFamilies{};
		std::ranges::move(
			queueFamiliesIndex
			| std::views::filter(isSuitable)
			| std::views::transform(toQueueFamily)
			, std::back_inserter(queueFamilies)
		);
		return queueFamilies;
	}
	[[nodiscard]] auto getSwapChainImages(const vk::Device& device, const vk::SwapchainKHR& swapChain) -> std::vector<vk::Image>
	{
		const auto Images = device.getSwapchainImagesKHR(swapChain);
		return std::vector<vk::Image>{};
	}
}

VulkanApplication::VulkanApplication()
	: window{ nullptr }
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
auto VulkanApplication::initInstance() -> void
{
	const Names layer{ getLayer() };
	const Names instanceExtension{ getInstanceExtension() };
	vk::ApplicationInfo applicationInfo{ "Hello Triangle" , VK_MAKE_VERSION(1, 0, 0) , "No Engine" , VK_MAKE_VERSION(1, 0, 0) , VK_API_VERSION_1_0 };
	vk::InstanceCreateInfo instanceCreateInfo{ {}, &applicationInfo, layer.getProxy(), instanceExtension.getProxy() };
	if (enableValidationLayers) instanceCreateInfo.setPNext(&debugMessengerCreateInfo); // extend create info with debug messenger
	instance = vk::createInstance(instanceCreateInfo);
}
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
auto VulkanApplication::initPhysicalDevice() -> void
{
	Names deviceExtension{ getDeviceExtension() };
	const auto isSuitable = [&, this](const vk::PhysicalDevice& physicalDevice)
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
auto VulkanApplication::initDevice() -> void
{
	const vk::PhysicalDeviceFeatures physicalDeviceFeatures{};
	Names layer{ getLayer() };
	Names deviceExtension{ getDeviceExtension() };
	const auto isSuitable = [&](QueueFamilyIndex i)
	{
		const auto isGraphical = physicalDevice.getQueueFamilyProperties()[i].queueFlags & vk::QueueFlagBits::eGraphics;
		const auto isSurfaceSupported = physicalDevice.getSurfaceSupportKHR(i, surface); // for presentation
		return isGraphical && isSurfaceSupported;
	};
	const auto toQueueCreateInfo = [](const auto& queueFamily) {
		const auto& [queueFamilyIndex, queuePriorities] = queueFamily;
		return vk::DeviceQueueCreateInfo{ {}, queueFamilyIndex, queuePriorities };
	};
	std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos{};
	std::ranges::move(getQueueFamilies(physicalDevice, isSuitable) | std::views::transform(toQueueCreateInfo), std::back_inserter(queueCreateInfos));
	vk::DeviceCreateInfo deviceCreateInfo{ {}, queueCreateInfos, layer.getProxy(), deviceExtension.getProxy(), &physicalDeviceFeatures };
	device = physicalDevice.createDevice(deviceCreateInfo);
}
auto VulkanApplication::initQueue() -> void
{
	int QueueFamilyIndex = 0; // TODO: always pick the first one tentatively for now
	int queueIndex = 0; // TODO: always pick the first one tentatively for now
	queue = device.getQueue(QueueFamilyIndex, queueIndex);
}
auto VulkanApplication::initSwapChain() -> void
{
	const auto surfaceAttributes = getSurfaceAttributes(surface, physicalDevice);
	const auto surfaceFormat = getSuitableSurfaceFormat(surfaceAttributes.formats);
	const auto presentMode = getSuitablePresentMode(surfaceAttributes.presentModes);
	const auto surfaceExtent = getSuitableSurfaceExtent(window, surfaceAttributes.capabilities);
	auto imageCount = surfaceAttributes.capabilities.minImageCount + 1;
	if (surfaceAttributes.capabilities.maxImageCount != 0) // not infinite
	{
		imageCount = std::min(imageCount, surfaceAttributes.capabilities.maxImageCount);
	}
	const auto isSuitableQueueFamilyIndex = [&](QueueFamilyIndex i)
	{
		const auto isGraphical = physicalDevice.getQueueFamilyProperties()[i].queueFlags & vk::QueueFlagBits::eGraphics;
		const auto isSurfaceSupported = physicalDevice.getSurfaceSupportKHR(i, surface); // for presentation
		return isGraphical && isSurfaceSupported;
	};
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
		, getQueueFamilies(physicalDevice, isSuitableQueueFamilyIndex).front().first // get the first queue family index
		, surfaceAttributes.capabilities.currentTransform
		, vk::CompositeAlphaFlagBitsKHR::eOpaque
		, presentMode
		, VK_TRUE
		, VK_NULL_HANDLE
	};
	swapChain = device.createSwapchainKHR(createInfo);
}
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
	const auto images = getSwapChainImages(device, swapChain);
	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();
	}
}




