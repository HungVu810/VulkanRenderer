#include <ranges>
#include <stdexcept>
#include <cassert>
#include <cstdlib>
#include <memory>
#include <algorithm>
#include <cstring>
#include <array>
#include <functional>
#include <fstream>
#include <filesystem>
#include <cstdint> // Needed for uint32_t
#include <limits> // Needed for std::numeric_limits
#include <algorithm> // Needed for std::clamp
#include "VulkanApplication.h"
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace
{
	// Class/Structs
	// Return a container of cstr proxies from names, which is used by the Vulkan API
	[[nodiscard]] auto getNamesProxy(const std::vector<std::string>& names) noexcept
	{
		const auto toRawString = [](std::string_view str) { return str.data(); };
		auto proxies = names | std::views::transform(toRawString) | std::ranges::to<std::vector>();
		return proxies;
	}
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
	[[nodiscard]] auto getEnabledLayersName()
	{
		auto strings = std::vector<std::string>{};
		if (!isValidationLayersEnabled) return strings;
		const auto validationLayerName = std::string{ "VK_LAYER_KHRONOS_validation" };
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
			const auto deviceProperties = vk::PhysicalDeviceProperties{ physicalDevice.getProperties() };
			const auto deviceFeatures = vk::PhysicalDeviceFeatures{ physicalDevice.getFeatures() };
			const auto getExtensionName = [](const vk::ExtensionProperties& properties) { return static_cast<std::string_view>(properties.extensionName); };
			const auto supportedDeviceExtensionsName = physicalDevice.enumerateDeviceExtensionProperties() | std::views::transform(getExtensionName);
			const auto surfaceAttributes = getSurfaceAttributes(surface, physicalDevice);
			const auto isSupported = [&](const std::string_view& deviceExtensionName) { return std::ranges::find(supportedDeviceExtensionsName, deviceExtensionName) != supportedDeviceExtensionsName.end(); };
			return deviceFeatures.geometryShader
				&& std::ranges::all_of(deviceExtensionsName, isSupported)
				&& !surfaceAttributes.formats.empty()
				&& !surfaceAttributes.presentModes.empty();
		};
		auto suitablePhysicalDevices = physicalDevices | std::views::filter(isSuitable);
		if (suitablePhysicalDevices.empty()) throw std::runtime_error{ "Can't find a suitable GPU." };
		const auto isDiscrete = [](const vk::PhysicalDevice& physicalDevice) { return physicalDevice.getProperties().deviceType == vk::PhysicalDeviceType::eDiscreteGpu; };
		const auto discretePhysicalDeviceIter = std::ranges::find_if(suitablePhysicalDevices, isDiscrete);
		if (discretePhysicalDeviceIter == suitablePhysicalDevices.end())
		{
			std::cout << tag::warning << "Using a non-discrete device for rendering.\n";
			return suitablePhysicalDevices.front();
		}
		else return *discretePhysicalDeviceIter;
	}
	[[nodiscard]] auto getSuitableSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& surfaceFormat) noexcept
	{
		const auto isSuitable = [](const vk::SurfaceFormatKHR& surfaceFormat)
		{
			return surfaceFormat.format == vk::Format::eB8G8R8A8Srgb
				&& surfaceFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear;
		};
		const auto surfaceFormatIter = std::ranges::find_if(surfaceFormat, isSuitable);
		if (surfaceFormatIter == surfaceFormat.end()) return surfaceFormat.at(0);
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
	[[nodiscard]] auto getSuitableQueueFamilies(const vk::PhysicalDevice& physicalDevice, vk::SurfaceKHR& surface) noexcept
	{
		const auto queueFamiliesProperties = physicalDevice.getQueueFamilyProperties();
		const auto queueFamiliesIndex = std::views::iota(0u, queueFamiliesProperties.size());
		const auto isSuitable = [&](QueueFamilyIndex i)
		{
			const auto isGraphical = physicalDevice.getQueueFamilyProperties()[i].queueFlags & vk::QueueFlagBits::eGraphics;
			const auto isSurfaceSupported = physicalDevice.getSurfaceSupportKHR(i, surface); // for presentation
			return isGraphical && isSurfaceSupported;
		};
		const auto toQueueFamily = [](QueueFamilyIndex i)
		{
			return QueueFamily{ i, { 1.0f } }; // One queue of priority level 1
		};
		auto queueFamilies = queueFamiliesIndex | std::views::filter(isSuitable) | std::views::transform(toQueueFamily) | std::ranges::to<std::vector>();
		return queueFamilies;
	}
	[[nodiscard]] auto getSwapChainImages(const vk::Device& device, const vk::SwapchainKHR& swapchain) noexcept
	{
		const auto images = device.getSwapchainImagesKHR(swapchain);
		return images;
	}
	[[nodiscard]] auto getShaderFile(const std::string& path)
	{
		// TODO: normalize file path?
		const auto filePath = std::filesystem::path{ path };
		auto file = std::ifstream{filePath , std::ios::binary};
		if (!file.is_open()) throw std::runtime_error(std::string{"Can't open file at "} + filePath.string());
		auto shaderFile = std::vector<char>(std::filesystem::file_size(filePath));
		file.read(shaderFile.data(), shaderFile.size());
		return shaderFile;
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
	, surfaceFormat{}
	, surfaceExtent{}
	, swapchain{}
	, imageViews{}
	, renderPass{}
	, pipelineLayout{}
	, graphicPipeline{}
{
}
VulkanApplication::~VulkanApplication()
{
	const auto destroy = [&](const auto& imageView)
	{
		device.destroyImageView(imageView);
	};
	std::ranges::for_each(imageViews, destroy);
	device.destroyRenderPass(renderPass);
	device.destroyPipeline(graphicPipeline);
	device.destroyPipelineLayout(pipelineLayout);
	device.destroySwapchainKHR(swapchain);
	device.destroy();
	if (isValidationLayersEnabled)
	{
		instance.destroyDebugUtilsMessengerEXT(debugMessenger);
	}
	instance.destroySurfaceKHR(surface);
	instance.destroy();
	glfwDestroyWindow(window);
	glfwTerminate();
}

void VulkanApplication::run() noexcept
{
	try
	{
		initWindow(); // Must be before initVulkan()
		initVulkan();
		mainLoop();
	}
	catch (const std::exception& except)
	{
		std::cerr << tag::exception << except.what() << std::endl;
	}
}

void VulkanApplication::initWindow()
{
	assertm(glfwInit() == GLFW_TRUE, "Failed to initalize GLFW");
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // Do not create an OpenGL context
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
	window = glfwCreateWindow(WIDTH, HEIGHT, "MyWindow", nullptr, nullptr);
	assertm(window, "Can't create window");
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
	assertm(result == VK_SUCCESS, "Failed to create window surface");
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
	const auto queueFamilyIndex = int{ 0 }; // TODO: always pick the first one tentatively for now
	const auto queueIndex = int{ 0 }; // TODO: always pick the first one tentatively for now
	queue = device.getQueue(queueFamilyIndex, queueIndex);
}
void VulkanApplication::initSwapChain()
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
	const auto queueFamilies{ getSuitableQueueFamilies(physicalDevice, surface) };
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
		, queueFamilies.front().first // get the first queue family index
		, surfaceAttributes.capabilities.currentTransform
		, vk::CompositeAlphaFlagBitsKHR::eOpaque
		, presentMode
		, VK_TRUE
		, VK_NULL_HANDLE
	};
	swapchain = device.createSwapchainKHR(createInfo);
}
void VulkanApplication::initImageViews()
{
	const auto images = getSwapChainImages(device, swapchain);
	const auto toImageViewCreateInfo = [&](const auto& image)
	{
		return vk::ImageViewCreateInfo{
			{}
			, image
			, vk::ImageViewType::e2D
			, surfaceFormat.format
			, {vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity}
			, {vk::ImageAspectFlagBits::eColor, 0u, 1u, 0u, 1u}
		};
	};
	const auto toImageView = [&](const auto& imageViewCreateInfo)
	{
		return device.createImageView(imageViewCreateInfo);
	};
	imageViews = images | std::views::transform(toImageViewCreateInfo) | std::views::transform(toImageView) | std::ranges::to<std::vector>();
}
void VulkanApplication::initRenderPass()
{
	// Attachment rule when attaching a framebuffer in the swapchain to the viewport
	const auto colorAttachmentDescription = vk::AttachmentDescription{
		{}
		, surfaceFormat.format
		, vk::SampleCountFlagBits::e1 // One sample, no multisampling
		, vk::AttachmentLoadOp::eClear // Clear the screen before attaching the framebuffer & depth buffer
		, vk::AttachmentStoreOp::eStore // Store the rendered framebuffer & depth buffer to memory after rendered
		, vk::AttachmentLoadOp::eDontCare // Stencil?
		, vk::AttachmentStoreOp::eDontCare // Stencil?
		, vk::ImageLayout::eUndefined // pre rendering
		, vk::ImageLayout::ePresentSrcKHR // post rendering, when the framebuffer is ready for attachment
	};
	// Subpass, only one which is the renderpass itself
	const auto colorAttachmentReference = vk::AttachmentReference{
		0 // description index in the descriptions array
		, vk::ImageLayout::eColorAttachmentOptimal
	};
	const auto subpassDescription = vk::SubpassDescription{
		{}
		, vk::PipelineBindPoint::eGraphics
		, {}
		, colorAttachmentReference
	};

	const auto renderPassCreateInfo = vk::RenderPassCreateInfo{
		{}
		, colorAttachmentDescription
		, subpassDescription
	};
	renderPass = device.createRenderPass(renderPassCreateInfo);
}
void VulkanApplication::initGraphicPipeline()
{
	// Vertex shader stage
	const auto vertexShaderFile = getShaderFile("Binaries/Shader/Vertex.spv");
	const auto vertexShaderModuleCreateInfo = vk::ShaderModuleCreateInfo{ {}, vertexShaderFile.size(), (uint32_t*)(vertexShaderFile.data()) };
	const auto vertexShaderModule = device.createShaderModule(vertexShaderModuleCreateInfo);
	const auto vertexShaderStageCreateInfo = vk::PipelineShaderStageCreateInfo{
		{}
		, vk::ShaderStageFlagBits::eVertex
		, vertexShaderModule
		, "main" // entry point
	};
	// Fragment shader stage
	const auto fragmentShaderFile = getShaderFile("Binaries/Shader/Fragment.spv");
	const auto fragmentCreateInfo = vk::ShaderModuleCreateInfo{ {}, fragmentShaderFile.size(), (uint32_t*)(fragmentShaderFile.data()) };
	const auto fragmentShaderModule = device.createShaderModule(fragmentCreateInfo);
	const auto fragmentShaderStageCreateInfo = vk::PipelineShaderStageCreateInfo{
		{}
		, vk::ShaderStageFlagBits::eFragment
		, fragmentShaderModule
		, "main" // entry point
	};
	// Shader stages
	const auto shaderStagesCreateInfo = std::vector<vk::PipelineShaderStageCreateInfo>{
		vertexShaderStageCreateInfo
		, fragmentShaderStageCreateInfo
	};

	// --------- FIXED-STATES
	// Support with vertex shader's input
	const auto vertexInputStateCreateInfo = vk::PipelineVertexInputStateCreateInfo{};
	// Support with vertex shader's output
	const auto inputAssemblyStateCreateInfo = vk::PipelineInputAssemblyStateCreateInfo{
		{}
		, vk::PrimitiveTopology::eTriangleList,
	};
	// Viewport
	const auto viewport = vk::Viewport{
		0.0f
		, 0.0f
		, static_cast<float>(surfaceExtent.width)
		, static_cast<float>(surfaceExtent.height)
		, 0.0f
		, 1.0f
	};
	const auto scissor = vk::Rect2D{ {0, 0}, surfaceExtent };
	const auto viewportStateCreateInfo = vk::PipelineViewportStateCreateInfo{
		{}
		, viewport
		, scissor
	};
	// Rasterizer
	const auto rasterizationStateCreateInfo = vk::PipelineRasterizationStateCreateInfo{
		{}
		, VK_FALSE
		, VK_FALSE
		, vk::PolygonMode::eFill
		, vk::CullModeFlagBits::eBack // cull back face
		, vk::FrontFace::eClockwise // the front face direction
		, VK_FALSE // influence the depth?
		, 0.0f
		, 0.0f
		, 0.0f
		, 1.0f // fragment line thickness
	};
	// Multisampling for anti-aliasing
	const auto multisampleStateCreateInfo = vk::PipelineMultisampleStateCreateInfo{
		{}
		, vk::SampleCountFlagBits::e1
		, VK_FALSE
	};
	// Depth and stencil testing
	const auto depthStencilStateCreateInfo = vk::PipelineDepthStencilStateCreateInfo{};
	// Color blending, mix the fragment's color value with the value in the framebuffer (if already existed)
	// Config per attached framebuffer
	const auto colorBlendAttachmentState = vk::PipelineColorBlendAttachmentState{
		VK_FALSE
		, vk::BlendFactor::eOne // Fragment's color
		, vk::BlendFactor::eZero // Color in the framebuffer
		, vk::BlendOp::eAdd
		, vk::BlendFactor::eOne // Fragment's alpha
		, vk::BlendFactor::eZero // Alpha in the framebuffer
		, vk::BlendOp::eAdd
		, vk::ColorComponentFlagBits::eR
			| vk::ColorComponentFlagBits::eB
			| vk::ColorComponentFlagBits::eG
			| vk::ColorComponentFlagBits::eA
	};
	// Global color blending settings
	const auto colorBlendStateCreateInfo = vk::PipelineColorBlendStateCreateInfo{
		{}
		, VK_FALSE
		, vk::LogicOp::eCopy
		, colorBlendAttachmentState
	};
	// --------- FIXED-STATES

	// Dynamic state, used to modify a subset of options of the fixed states without recreating the pipeline
	const auto dynamicStateCreateInfo = vk::PipelineDynamicStateCreateInfo{};

	// Pipeline layout, for assigning uniform values to shaders
	const auto pipelineLayoutCreateInfo = vk::PipelineLayoutCreateInfo{};
	pipelineLayout = device.createPipelineLayout(pipelineLayoutCreateInfo);

	// Graphic pipeline
	const auto pipelineCreateInfo = vk::GraphicsPipelineCreateInfo{
		{}
		, shaderStagesCreateInfo
		, &vertexInputStateCreateInfo
		, &inputAssemblyStateCreateInfo
		, VK_NULL_HANDLE
		, &viewportStateCreateInfo
		, &rasterizationStateCreateInfo
		, &multisampleStateCreateInfo
		, &depthStencilStateCreateInfo
		, &colorBlendStateCreateInfo
		, &dynamicStateCreateInfo
		, pipelineLayout
		, renderPass
		, 0 // Index of the subpass
		// The next 2 params is used to create multiples pipelines with
		// multiple pipeline create infos in one single call
	};
	const auto resultValue = device.createGraphicsPipeline(VK_NULL_HANDLE, pipelineCreateInfo); // Can cache the pipeline in the VK_NULL_HANDLE argument
	if (resultValue.result != vk::Result::eSuccess) throw std::runtime_error{ "Failed to create a graphic pipeline" };
	graphicPipeline = resultValue.value;

	device.destroyShaderModule(vertexShaderModule);
	device.destroyShaderModule(fragmentShaderModule);
}

void VulkanApplication::initVulkan()
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
	initImageViews();
	initRenderPass();
	initGraphicPipeline();
}

void VulkanApplication::mainLoop()
{
	//const auto images = getSwapChainImages(device, swapchain);

	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();
	}
}




