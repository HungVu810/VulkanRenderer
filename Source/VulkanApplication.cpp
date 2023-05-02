#include <ranges>
#include <algorithm>
#include <stdexcept>
#include <cstdint> // Needed for uint32_t
#include <limits> // Needed for std::numeric_limits
#include "VulkanApplication.h" // Do not place this right above the VULKAN_HPP_DEFAULT macro
#include "Geometry.h"
#include "Shader.h"
#include "Utilities.h"
#include "Allocation.h"
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

// Volume data extraction
#include <fstream>
#include <filesystem>
#include <iostream>
#include <vector>
#include <span>
#include <cassert>

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
	[[nodiscard]] auto getSuitableSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& surfaceFormat) noexcept
	{
		const auto isSuitable = [](const vk::SurfaceFormatKHR& surfaceFormat)
		{
			return surfaceFormat.format == format::Image
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
	[[nodiscard]] auto getSuitableQueueFamiliesIndex(const vk::PhysicalDevice& physicalDevice, vk::SurfaceKHR& surface)
	{
		const auto queueFamilies = getSuitableQueueFamilies(physicalDevice, surface);
		const auto toQueueFamilyIndex = [](const QueueFamily& queueFamily){return queueFamily.first;};
		const auto queueFamiliesIndex = queueFamilies | std::views::transform(toQueueFamilyIndex) | std::ranges::to<std::vector>();
		return queueFamiliesIndex;
	}
	[[nodiscard]] auto inline checkFormatFeatures(const vk::PhysicalDevice& physicalDevice, vk::Format format, vk::FormatFeatureFlagBits requestedFormatFeatures)
	{
		const auto supportedFormatFeatures = physicalDevice.getFormatProperties(format).optimalTilingFeatures;
		const auto isSupported = supportedFormatFeatures & requestedFormatFeatures;
		if (!isSupported) throw std::runtime_error{"Requested format features are not supported"};
	}

	// TODO: Volume data extraction, only needed by the shader?
	[[nodiscard]] inline auto getIntensity(std::span<Intensity, NUM_INTENSITIES> slides, int z, int y, int x)
	{
		return slides[(z * SLIDE_HEIGHT * SLIDE_WIDTH) + (y * SLIDE_WIDTH) + x];
	}
	void setVolumeData(std::vector<Intensity>& intensities)
	{
		// TODO: intensities(NUM_INTENSITY), allocate upfront
		for (int i = 1; i <= NUM_SLIDES; i++)
		{
			const auto path = std::filesystem::path{VOLUME_DATA"/CThead." + std::to_string(i)};
			auto ctFile = std::ifstream{path, std::ios_base::binary};
			if (!ctFile) throw std::runtime_error{"Can't open file at " + path.string()};

			auto intensity = uint16_t{0};
			while (ctFile.read(reinterpret_cast<char*>(&intensity), sizeof(intensity)))
			{
				// Swap byte order if running on little-endian system
				#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
					data = (data >> 8) | (data << 8);
				#endif
				intensities.push_back(intensity);
			}
		}
		assert(intensities.size() == NUM_INTENSITIES); // Sanity check
	}
	//auto getRaycastViewportImage(uint32_t width, uint32_t height)
	//{
	//	// TODO: use cartesian product instead
	//	const auto widthIndices = std::views::iota(0U, WIDTH);
	//	const auto heightIndices = std::views::iota(0U, WIDTH);
	//}
}

VulkanApplication::VulkanApplication()
	: importVolumeDataWorker{}
	, intensities{}

	, window{nullptr}
	, validateShadersWorker{}
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
	, swapchainImageViews{}
	, renderPass{}
	, shaderMap{}
	, vertexBuffer{}
	, pipelineLayout{}
	, graphicPipeline{}
	, framebuffers{}
	, commandPool{}
	, commandBuffers{}
	, isAcquiredImageReadSemaphore{}
	, isImageRenderedSemaphore{}
	, isCommandBufferExecutedFence{}
{
}

VulkanApplication::~VulkanApplication() noexcept {}

void VulkanApplication::run() noexcept
{
	try
	{
		validateShadersWorker = std::thread{validateShaders, std::ref(shaderMap)};
		validateShadersWorker.join();
		importVolumeDataWorker = std::thread{setVolumeData, std::ref(intensities)};
		initWindow(); // Must be before initVulkan()
		initVulkan();
		mainLoop();
		cleanUp(); // Can't be put in the class' destructor due to potential exceptions
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
	const auto queueFamilyIndex = int{ 0 }; // TODO: always pick the first one tentatively for now
	const auto queueIndex = int{ 0 }; // TODO: always pick the first one tentatively for now
	queue = device.getQueue(queueFamilyIndex, queueIndex);
}
void VulkanApplication::initSwapChain()
{
	const auto surfaceAttributes = getSurfaceAttributes(surface, physicalDevice);
	if (surfaceAttributes.capabilities.supportedUsageFlags == vk::ImageUsageFlagBits::eStorage) throw std::runtime_error{"Surface image can't be used as a storage"};

	surfaceFormat = getSuitableSurfaceFormat(surfaceAttributes.formats);
	checkFormatFeatures(physicalDevice, surfaceFormat.format, vk::FormatFeatureFlagBits::eStorageImage);

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
		, 1 // Number of layers per image, more than 1 for stereoscopic application
		, vk::ImageUsageFlagBits::eTransferDst // | vk::ImageUsageFlagBits::eColorAttachment (original)
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

// Default
void VulkanApplication::initImageViews()
{
	const auto images = device.getSwapchainImagesKHR(swapchain);
	const auto toImageView = [&](const vk::Image& image)
	{
		const auto imageViewCreateInfo = vk::ImageViewCreateInfo{
			{}
			, image
			, vk::ImageViewType::e2D
			, surfaceFormat.format
			, {vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity}
			, {vk::ImageAspectFlagBits::eColor, 0u, 1u, 0u, 1u}
		};
		return device.createImageView(imageViewCreateInfo);
	};
	swapchainImageViews = images | std::views::transform(toImageView) | std::ranges::to<std::vector>();
}
void VulkanApplication::initRenderPass()
{
	// Descripes how the image views are load/store
	const auto attachmentDescription = vk::AttachmentDescription{
		{}
		, surfaceFormat.format
		, vk::SampleCountFlagBits::e1 // One sample, no multisampling
		, vk::AttachmentLoadOp::eClear
		, vk::AttachmentStoreOp::eStore
		, vk::AttachmentLoadOp::eDontCare
		, vk::AttachmentStoreOp::eDontCare
		, vk::ImageLayout::eUndefined // Pre renderpass instance
		, vk::ImageLayout::ePresentSrcKHR // Post renderpass instance
	};

	// Subpass description
	const auto attachmentIndex = 0U;
	const auto attachmentReference = vk::AttachmentReference{
		attachmentIndex
		, vk::ImageLayout::eColorAttachmentOptimal
	};
	const auto subpassDescription = vk::SubpassDescription{
		{}
		, vk::PipelineBindPoint::eGraphics
		, {}
		, attachmentReference
	};

	// Dependency
	const auto subpassDependency = vk::SubpassDependency{
		VK_SUBPASS_EXTERNAL // The other subpass
		, 0U // This subpass
		, vk::PipelineStageFlagBits::eColorAttachmentOutput
		, vk::PipelineStageFlagBits::eColorAttachmentOutput
		, vk::AccessFlagBits::eNone
		, vk::AccessFlagBits::eColorAttachmentWrite
	};

	const auto renderPassCreateInfo = vk::RenderPassCreateInfo{
		{}
		, attachmentDescription
		, subpassDescription
		, subpassDependency
	};
	renderPass = device.createRenderPass(renderPassCreateInfo);
}
void VulkanApplication::initGraphicPipeline()
{
	// Vertex shader stage
	const auto vertexShaderBinaryData = getShaderBinaryData(shaderMap, "General.vert");
	const auto vertexShaderModuleCreateInfo = vk::ShaderModuleCreateInfo{{}, vertexShaderBinaryData};
	const auto vertexShaderModule = device.createShaderModule(vertexShaderModuleCreateInfo);
	const auto vertexShaderStageCreateInfo = vk::PipelineShaderStageCreateInfo{
		{}
		, vk::ShaderStageFlagBits::eVertex
		, vertexShaderModule
		, "main" // entry point
	};
	// Fragment shader stage
	const auto fragmentShaderBinaryData = getShaderBinaryData(shaderMap, "General.frag");
	const auto fragmentShaderModuleCreateInfo = vk::ShaderModuleCreateInfo{{}, fragmentShaderBinaryData};
	const auto fragmentShaderModule = device.createShaderModule(fragmentShaderModuleCreateInfo);
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
	// NDC space
	const auto bindingNumber = 0U;
	const auto triangle = std::vector<Vertex>{
		{{0.5f, 0.5f, 0.0f}, {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}}
		, {{-0.5f, 0.5f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}}
		, {{0.0f, -0.5f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}}
	};
	// Vertex input state
	const auto vertexInputBindingDescription = vk::VertexInputBindingDescription{
		bindingNumber
		, sizeof(Vertex)
	};
	const auto vertexInputAttributeDescription = std::vector<vk::VertexInputAttributeDescription>{
		 {0, bindingNumber, format::vec3, offsetof(Vertex, position)}
		 , {1, bindingNumber, format::vec3, offsetof(Vertex, normal)}
		 , {2, bindingNumber, format::vec3, offsetof(Vertex, color)}
	};

	const auto vertexInputStateCreateInfo = vk::PipelineVertexInputStateCreateInfo{
		{}
		, vertexInputBindingDescription
		, vertexInputAttributeDescription
	};

	const auto queueFamiliesIndex = getSuitableQueueFamiliesIndex(physicalDevice, surface);
	const auto bufferCreateInfo = vk::BufferCreateInfo{
		{}
		, triangle.size() * sizeof(Vertex)
		, vk::BufferUsageFlagBits::eVertexBuffer
		, vk::SharingMode::eExclusive
		, queueFamiliesIndex

	};
	vertexBuffer = device.createBuffer(bufferCreateInfo);
	const auto memoryRequirements = device.getBufferMemoryRequirements(vertexBuffer);
	vertexBufferMemory = allocateMemory(device, physicalDevice, memoryRequirements, true);
	device.bindBufferMemory(vertexBuffer, vertexBufferMemory, 0);
	void* memory = device.mapMemory(vertexBufferMemory, 0, bufferCreateInfo.size); // Map physical memory to logical memory
	std::memcpy(memory, triangle.data(), bufferCreateInfo.size);
	device.unmapMemory(vertexBufferMemory);

	// Input Assembly
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

	// Assigning uniform values to shaders
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
		, 0 // Index of the renderpass' subpass that will uses this pipeline
		// The next 2 params is used to create new pipelines with multiple
		// create infos in one single call
	};
	const auto resultValue = device.createGraphicsPipeline(VK_NULL_HANDLE, pipelineCreateInfo); // Can cache the pipeline in the VK_NULL_HANDLE argument
	if (resultValue.result != vk::Result::eSuccess) throw std::runtime_error{ "Failed to create a graphic pipeline" };
	graphicPipeline = resultValue.value;

	device.destroyShaderModule(vertexShaderModule);
	device.destroyShaderModule(fragmentShaderModule);
}
void VulkanApplication::initFrameBuffer()
{
	framebuffers.resize(swapchainImageViews.size());
	const auto toFramebuffer = [&](const vk::ImageView& imageView)
	{
		const auto framebufferCreateInfo = vk::FramebufferCreateInfo{
			{}
			, renderPass
			, imageView // Attachments
			, surfaceExtent.width
			, surfaceExtent.height
			, 1 // Single layer, not doing stereo-rendering
		};
		return device.createFramebuffer(framebufferCreateInfo);
	};
	framebuffers = swapchainImageViews | std::views::transform(toFramebuffer) | std::ranges::to<std::vector>();
}
// Default

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

// ********* Volume Rendering *********, DONT NEED A FRAMEBUFFER, WE USE THE SWAPCHAIN DIRECTLY SINCE NO RENDERPASS IS MADE
// DONT NEED AN SWAPCHAIN IMAGE VIEW AS WELL
void VulkanApplication::initComputePipeline()
{
	importVolumeDataWorker.join(); // Make sure the volume data is ready

	// Format check
	checkFormatFeatures(physicalDevice, surfaceFormat.format, vk::FormatFeatureFlagBits::eStorageImage);
	checkFormatFeatures(physicalDevice, surfaceFormat.format, vk::FormatFeatureFlagBits::eTransferDst);
	checkFormatFeatures(physicalDevice, surfaceFormat.format, vk::FormatFeatureFlagBits::eTransferSrc);
	//formatFeatures = physicalDevice.getFormatProperties(format::RenderTarget).optimalTilingFeatures; // This format is for the raycasted image
	//assert(formatFeatures & (vk::FormatFeatureFlagBits::eStorageImage | vk::FormatFeatureFlagBits::eTransferSrc));

	const auto queueFamiliesIndex = getSuitableQueueFamiliesIndex(physicalDevice, surface);

	// Reserve memory for the raycasted image
	const auto raycastedImageCreateInfo = vk::ImageCreateInfo{
		{}
		, vk::ImageType::e2D
		, surfaceFormat.format
		, vk::Extent3D{surfaceExtent, 1}
		, 1 // The only mip level for this image
		, 1 // Single layer, no stereo-rendering
		, vk::SampleCountFlagBits::e1 // One sample, no multisampling
		, vk::ImageTiling::eOptimal
		, vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc // Transfer command from the staging device memory to the image memory via cmdCopy
		, vk::SharingMode::eExclusive
		, queueFamiliesIndex
	};
	raycastedImage = device.createImage(raycastedImageCreateInfo);
	const auto raycastedImageMemoryRequirements = device.getImageMemoryRequirements(raycastedImage);
	raycastedImageMemory = allocateMemory(device, physicalDevice, raycastedImageMemoryRequirements, false); // TODO: WHY DOESN'T THIS ALLOW HOST-VISIBLE MEMORY?
	device.bindImageMemory(raycastedImage, raycastedImageMemory, 0);

	//// TODO: using Intensities = std::pair<std::vector<uint64_t>, format::Short>
	//// TODO: return {stagingBuffer, saginBufferMemory} -> [resource, resourceMemory]
	//const auto imageIndex = 0;
	//const auto swapchainImage = device.getSwapchainImagesKHR(swapchain)[imageIndex];
	//// Reserve the raycasted target image resource for the compute shader
	//const auto raycastedImageMemoryRequirements = device.getImageMemoryRequirements(raycastedImage);
	//raycastedImageMemory = allocateMemory(device, physicalDevice, raycastedImageMemoryRequirements, false);
	//device.bindImageMemory(raycastedImage, raycastedImageMemory, 0);

	// Reserve device memory for volume image
	const auto volumeImageCreateInfo = vk::ImageCreateInfo{
		{}
		, vk::ImageType::e3D
		, format::Short // Type of Intensity
		, {SLIDE_WIDTH, SLIDE_HEIGHT, NUM_SLIDES}
		, 1 // The only mip level for this image
		, 1 // Single layer, no stereo-rendering
		, vk::SampleCountFlagBits::e1 // One sample, no multisampling
		, vk::ImageTiling::eOptimal
		, vk::ImageUsageFlagBits::eSampled // Sampled by the compute shader
			| vk::ImageUsageFlagBits::eTransferDst // Volumn data is transfered from a staging buffer transferSrc
		, vk::SharingMode::eExclusive
		, queueFamiliesIndex
	};
	volumeImage = device.createImage(volumeImageCreateInfo);
	const auto volumeImageMemoryRequirements = device.getImageMemoryRequirements(volumeImage);
	volumeImageMemory = allocateMemory(device, physicalDevice, volumeImageMemoryRequirements, false); // TODO: WHY DOESN'T THIS ALLOW HOST-VISIBLE MEMORY?
	device.bindImageMemory(volumeImage, volumeImageMemory, 0);

	// Reserve device memory for staging buffer and upload the data to the memory
	const auto stagingBufferCreateInfo = vk::BufferCreateInfo{
		{}
		, TOTAL_SCAN_BYTES
		, vk::BufferUsageFlagBits::eTransferSrc // Store and transfer volume data to volume image stored on the device memory via a copy command
		, vk::SharingMode::eExclusive
		, queueFamiliesIndex
	};
	stagingBuffer = device.createBuffer(stagingBufferCreateInfo);
	const auto stagingBufferMemoryRequirement = device.getBufferMemoryRequirements(stagingBuffer);
	stagingBufferMemory = allocateMemory(device, physicalDevice, stagingBufferMemoryRequirement, true); // TODO: WHY THIS ALLOW HOST-VISIBLE MEMORY?
	device.bindBufferMemory(stagingBuffer, stagingBufferMemory, 0);
	void* memory = device.mapMemory(stagingBufferMemory, 0, stagingBufferMemoryRequirement.size);
	std::memcpy(memory, intensities.data(), TOTAL_SCAN_BYTES);
	device.unmapMemory(stagingBufferMemory);

	// Create a descriptor for the volumn image and the raycasted image
	const auto maxDescriptorSets = 1;	
	const auto descriptorPoolSizes = std::vector<vk::DescriptorPoolSize>{
		{vk::DescriptorType::eSampledImage, 1}
		, {vk::DescriptorType::eStorageImage, 1}
		//, {vk::DescriptorType::eStorageBufferDynamic, 1}
	};
	const auto descriptorPoolCreateInfo = vk::DescriptorPoolCreateInfo{
		{}
		, maxDescriptorSets
		, descriptorPoolSizes
	};
	descriptorPool = device.createDescriptorPool(descriptorPoolCreateInfo);

	const auto samplerCreateInfo = vk::SamplerCreateInfo{};
	sampler = device.createSampler(samplerCreateInfo);
	const auto volumnImageBinding = vk::DescriptorSetLayoutBinding{
		0
		, vk::DescriptorType::eSampledImage
		, vk::ShaderStageFlagBits::eCompute
		, sampler
	};
	const auto raycastedImageBinding = vk::DescriptorSetLayoutBinding{
		1
		, vk::DescriptorType::eStorageImage
		, vk::ShaderStageFlagBits::eCompute
		, sampler
	};
	//const auto swapchainImageBinding = vk::DescriptorSetLayoutBinding{
	//	1
	//	, vk::DescriptorType::eStorageImage
	//	, vk::ShaderStageFlagBits::eCompute
	//	, sampler
	//};

	//const auto layoutBindings = {volumnImageBinding, swapchainImageBinding};
	const auto layoutBindings = {volumnImageBinding, raycastedImageBinding};
	const auto descriptorSetLayoutCreateInfo = vk::DescriptorSetLayoutCreateInfo{
		{}
		, layoutBindings
	};
	descriptorSetLayout = device.createDescriptorSetLayout(descriptorSetLayoutCreateInfo);

	const auto descriptorSetsAllocateInfo = vk::DescriptorSetAllocateInfo{descriptorPool, descriptorSetLayout};
	descriptorSets = device.allocateDescriptorSets(descriptorSetsAllocateInfo);

	// Bind the volume image view to the descriptor
	const auto imageSubresourceRange = vk::ImageSubresourceRange{
		vk::ImageAspectFlagBits::eColor
		, 0
		, 1
		, 0
		, 1
	};
	const auto volumeImageViewCreateInfo = vk::ImageViewCreateInfo{
		{}
		, volumeImage
		, vk::ImageViewType::e3D
		, format::Short // Type of Intensity
		, {vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity}
		, imageSubresourceRange
	};
	volumeImageView = device.createImageView(volumeImageViewCreateInfo);
	auto descriptorImageInfo = vk::DescriptorImageInfo{
		sampler
		, volumeImageView
		, vk::ImageLayout::eShaderReadOnlyOptimal // Expected layout of the descriptor
	};
	auto writeDescriptorSet = vk::WriteDescriptorSet{
		descriptorSets.front()
		, 0
		, 0
		, vk::DescriptorType::eSampledImage
		, descriptorImageInfo
	};
	device.updateDescriptorSets(writeDescriptorSet, {});

	// Bind the raycasted image view to the descriptor
	const auto raycastedImageViewCreateInfo = vk::ImageViewCreateInfo{
		{}
		, raycastedImage
		, vk::ImageViewType::e2D
		, surfaceFormat.format
		, {vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity}
		, imageSubresourceRange
	};
	raycastedImageView = device.createImageView(raycastedImageViewCreateInfo);
	descriptorImageInfo = vk::DescriptorImageInfo{
		sampler
		, raycastedImageView
		, vk::ImageLayout::eGeneral // Expected layout of the descriptor
	};
	writeDescriptorSet = vk::WriteDescriptorSet{
		descriptorSets.front()
		, 1
		, 0
		, vk::DescriptorType::eStorageImage
		, descriptorImageInfo
	};
	device.updateDescriptorSets(writeDescriptorSet, {});

	// Compute pipeline
	const auto volumeShaderBinaryData = getShaderBinaryData(shaderMap, "VolumeRendering.comp");
	const auto volumeShaderModuleCreateInfo = vk::ShaderModuleCreateInfo{{}, volumeShaderBinaryData};
	volumeShaderModule = device.createShaderModule(volumeShaderModuleCreateInfo);
	const auto computeShaderStageCreateInfo = vk::PipelineShaderStageCreateInfo{
		{}
		, vk::ShaderStageFlagBits::eCompute
		, volumeShaderModule
		, "main"
	};

	const auto pipelineLayoutCreateInfo = vk::PipelineLayoutCreateInfo{{}, descriptorSetLayout};
	computePipelineLayout = device.createPipelineLayout(pipelineLayoutCreateInfo);

	const auto computePipelineCreateInfo = vk::ComputePipelineCreateInfo{
		{}
		, computeShaderStageCreateInfo
		, computePipelineLayout
	};
	const auto resultValue = device.createComputePipeline({}, computePipelineCreateInfo);
	if (resultValue.result != vk::Result::eSuccess) throw std::runtime_error{"Unable to create a compute pipeline."};
	computePipeline = resultValue.value;
}
void VulkanApplication::transferStagingBufferToVolumeImageAndTransitionLayouts()
{
	// Temporary company buffer to transfer the data to the 3D image descriptor
	const auto queueFamilyIndex = getSuitableQueueFamiliesIndex(physicalDevice, surface).front();

	// Create a temporary commandBuffer to perform a copy command
	const auto commandPoolCreateInfo = vk::CommandPoolCreateInfo{
		vk::CommandPoolCreateFlagBits::eTransient
		, queueFamilyIndex
	};
	const auto commandPool = device.createCommandPool(commandPoolCreateInfo);
	const auto commandBufferAllocateInfo = vk::CommandBufferAllocateInfo{
		commandPool
		, vk::CommandBufferLevel::ePrimary
		, 1
	};
	const auto commandBuffers = device.allocateCommandBuffers(commandBufferAllocateInfo);
	const auto commandBuffer = commandBuffers.front();

	// Recording
	const auto commandBufferBeginInfo = vk::CommandBufferBeginInfo{
		vk::CommandBufferUsageFlagBits::eOneTimeSubmit
	};
	commandBuffer.begin(commandBufferBeginInfo);
	const auto copyRegion = vk::BufferImageCopy{
		0
		, 0
		, 0
		, vk::ImageSubresourceLayers{
			vk::ImageAspectFlagBits::eColor
			, 0
			, 0
			, 1
		}
		, vk::Offset3D{0, 0, 0}
		, vk::Extent3D{SLIDE_WIDTH, SLIDE_HEIGHT, NUM_SLIDES}
	};
	commandBuffer.copyBufferToImage(stagingBuffer, volumeImage, vk::ImageLayout::eTransferDstOptimal, copyRegion); // Volume image layout: undefined -> transferDstOptimal

	// Transition layout of the volume image to a read-only layout before read by the compute pipeline
	// Volume image layout: transferDstOptimal -> shaderReadOnlyOptimal
	const auto volumeImageBarrierToDescriptor = vk::ImageMemoryBarrier{
		vk::AccessFlagBits::eTransferWrite
		, vk::AccessFlagBits::eShaderRead
		, vk::ImageLayout::eTransferDstOptimal
		, vk::ImageLayout::eShaderReadOnlyOptimal
		, VK_QUEUE_FAMILY_IGNORED // Same queue family, don't transfer the queue ownership
		, VK_QUEUE_FAMILY_IGNORED
		, volumeImage
		, vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
	};
	commandBuffer.pipelineBarrier(
		vk::PipelineStageFlagBits::eTransfer // Copy command
		, vk::PipelineStageFlagBits::eComputeShader
		, {}
		, {}
		, {}
		, volumeImageBarrierToDescriptor
	);

	// Transition all swapchain image to presentable layout so it is valid to be accquired from the device
	// swapchainImages layout, undefined -> presentSrcKHR
	const auto swapchainImages = device.getSwapchainImagesKHR(swapchain);
	// Doesn't need to be synchronized, we change all of them simutaneously. But
	// because we are using imagebarrier and set to memoryread or memorywrite,
	// they will be in order? Probably not because the images are seperated resources
	for (const auto& image : swapchainImages)
	{
		const auto imageTransition = vk::ImageMemoryBarrier{
			vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite
			, vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite
			, vk::ImageLayout::eUndefined
			, vk::ImageLayout::ePresentSrcKHR
			, VK_QUEUE_FAMILY_IGNORED // Same queue family, don't transfer the queue ownership
			, VK_QUEUE_FAMILY_IGNORED
			, image
			, vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
		};
		commandBuffer.pipelineBarrier(
			vk::PipelineStageFlagBits::eAllCommands
			, vk::PipelineStageFlagBits::eAllCommands
			, {}
			, {}
			, {}
			, imageTransition
		);
	}
	commandBuffer.end();

	// Submit to transfer queue
	const auto isBufferTransferedFence = device.createFence({});
	queue.submit({}, isBufferTransferedFence);
	std::ignore = device.waitForFences(isBufferTransferedFence, VK_TRUE, UINT64_MAX); // Destroy when finished

	device.destroy(isBufferTransferedFence);
	device.freeCommandBuffers(commandPool, commandBuffer);
	device.destroyCommandPool(commandPool);
}
void VulkanApplication::recordVolumeCommandBuffer(vk::CommandBuffer& commandBuffer, uint32_t imageIndex, vk::Result fenceResult)
{
	const auto commandBufferBeginInfo = vk::CommandBufferBeginInfo{};
	commandBuffer.begin(commandBufferBeginInfo);
	
	// TDO:top or bottom of recording-> stage: allcommand, access: read

	// Transition layout of the image descriptors before compute pipeline
	// Raycasted image layout: undefined (default)/transferSrc -> general, expected by the descriptor
	const auto raycasteedImageBarrierToDescriptor = vk::ImageMemoryBarrier{
		vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite
		, vk::AccessFlagBits::eShaderWrite
		, fenceResult == vk::Result::eSuccess ? vk::ImageLayout::eUndefined : vk::ImageLayout::eTransferSrcOptimal
		, vk::ImageLayout::eGeneral
		, VK_QUEUE_FAMILY_IGNORED // Same queue family, don't transfer the queue ownership
		, VK_QUEUE_FAMILY_IGNORED
		, raycastedImage
		, vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
	};
	commandBuffer.pipelineBarrier(
		vk::PipelineStageFlagBits::eAllCommands
		, vk::PipelineStageFlagBits::eComputeShader
		, {}
		, {}
		, {}
		, raycasteedImageBarrierToDescriptor
	);

	commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, computePipeline);
	commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, computePipelineLayout, 0, descriptorSets, {}); // 3D volume data
	const auto numInvocationPerX = 10; // Also put this number in the compute shader
	const auto numInvocationPerY = 10; // Also put this number in the compute shader
	assert((WIDTH % numInvocationPerX) == 0 && (HEIGHT % numInvocationPerY) == 0);
	commandBuffer.dispatch(WIDTH / numInvocationPerX, HEIGHT / numInvocationPerY, 1); // *******NOTE: group size must be at least 1

	const auto swapchainImage = device.getSwapchainImagesKHR(swapchain)[imageIndex];

	// Barrier to sync writting to raycastedImage via compute shader and copy it to swapchain image
	// Raycasted image layout: general -> transferSrc, before copy commmand
	const auto raycastedImageToTransfer = vk::ImageMemoryBarrier{
		vk::AccessFlagBits::eShaderWrite // Compute shader writes to raycasted image, vk::AccessFlagBits::eShaderWrite, only use this when the shader write to the memory
		, vk::AccessFlagBits::eTransferRead // Wait until raycasted image is finished written to then copy
		, vk::ImageLayout::eGeneral
		, vk::ImageLayout::eTransferSrcOptimal
		, VK_QUEUE_FAMILY_IGNORED // Same queue family, don't transfer the queue ownership
		, VK_QUEUE_FAMILY_IGNORED // Same queue family, don't transfer the queue ownership
		, raycastedImage
		, vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
	};
	// Swapchain image layout: undefined (default)/presentSrcKHR -> transferDst, before copy command
	const auto swapchainImageToTransfer = vk::ImageMemoryBarrier{
		vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite
		, vk::AccessFlagBits::eTransferWrite // Wait until raycasted image is finished written to then copy
		, fenceResult == vk::Result::eSuccess ? vk::ImageLayout::eUndefined : vk::ImageLayout::ePresentSrcKHR
		, vk::ImageLayout::eTransferDstOptimal
		, VK_QUEUE_FAMILY_IGNORED // Same queue family, don't transfer the queue ownership
		, VK_QUEUE_FAMILY_IGNORED // Same queue family, don't transfer the queue ownership
		, swapchainImage
		, vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
	};
	commandBuffer.pipelineBarrier(
		vk::PipelineStageFlagBits::eAllCommands
		, vk::PipelineStageFlagBits::eTransfer
		, {}
		, {}
		, {}
		, {raycastedImageToTransfer, swapchainImageToTransfer}
	);

	commandBuffer.copyImage(
		raycastedImage
		, vk::ImageLayout::eTransferSrcOptimal
		, swapchainImage
		, vk::ImageLayout::eTransferDstOptimal
		, vk::ImageCopy{
			vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, 0, 0, 1}
			, {0, 0, 0}
			, vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, 0, 0, 1}
			, {0, 0, 0}
			, vk::Extent3D{surfaceExtent, 1}
		}
	);

	// Transfer the swapchain image layout back to a presentable layout
	// Swapchain image layout: transferDst -> presentSrcKHR, before presented
	const auto toPresentLayout = vk::ImageMemoryBarrier{
		vk::AccessFlagBits::eTransferWrite // written during the coppy command
		, vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite
		, vk::ImageLayout::eTransferDstOptimal
		, vk::ImageLayout::ePresentSrcKHR
		, VK_QUEUE_FAMILY_IGNORED
		, VK_QUEUE_FAMILY_IGNORED
		, swapchainImage
		, vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
	};
	commandBuffer.pipelineBarrier(
		vk::PipelineStageFlagBits::eTransfer
		, vk::PipelineStageFlagBits::eAllCommands
		, {}
		, {}
		, {}
		, toPresentLayout
	);

	commandBuffer.end();
}
void VulkanApplication::drawVolumeFrame()
{
	// IF THIS IS TOO HARD, CREAT A QUAD IMAGE THEN SAMPLE THE CONTENT IN graphic pipeline like sacsha willem

	const auto fenceResult = device.getFenceStatus(isCommandBufferExecutedFence); // Only signaled for the first loop

	// Fence and submission
	std::ignore = device.waitForFences(isCommandBufferExecutedFence, VK_TRUE, UINT64_MAX); // Avoid modifying the command buffer when it's in used by the device
	device.resetFences(isCommandBufferExecutedFence);

	const auto resultValue = device.acquireNextImageKHR(swapchain, UINT64_MAX, isAcquiredImageReadSemaphore); // Semaphore will be raised when the acquired image is finished reading by the engine
	if (resultValue.result != vk::Result::eSuccess) throw std::runtime_error{"Failed to acquire the next image index."};

	const auto imageIndex = resultValue.value;
	const auto waitStages = std::vector<vk::PipelineStageFlags>{vk::PipelineStageFlagBits::eComputeShader};
	recordVolumeCommandBuffer(commandBuffers.front(), imageIndex, fenceResult);
	const auto submitInfo = vk::SubmitInfo{
		isAcquiredImageReadSemaphore // Wait for the image to be finished reading, then we will modify it via the commands in the commandBuffers
		, waitStages // Wait only if we have inflight command buffers
		, commandBuffers
		, isImageRenderedSemaphore // Raise when finished executing the commands
	};
	queue.submit(submitInfo, isCommandBufferExecutedFence); // Raise when finished executing the commands

	// const auto imageSubresourceLayout = device.getImageSubresourceLayout(device.getSwapchainImagesKHR(swapchain)[imageIndex], vk::ImageSubresource{vk::ImageAspectFlagBits::eColor, 0, 0});
	// assert(imageSubresourceLayout. == vk::ImageLayout::ePresentSrcKHR);

	const auto presentInfoKHR = vk::PresentInfoKHR{
		isImageRenderedSemaphore
		, swapchain
		, imageIndex
	};
	const auto presentResult = queue.presentKHR(presentInfoKHR);
	if (presentResult != vk::Result::eSuccess) throw std::runtime_error{"Failed to present image."};

}
// ********* Volume Rendering *********

void VulkanApplication::initVulkan()
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
	initSwapChain();

	//initImageViews();
	//initRenderPass();
	//initGraphicPipeline();
	//initFrameBuffer();

	//// uncomment ------------
	//initVolumeRenderPass();
	initComputePipeline();
	transferStagingBufferToVolumeImageAndTransitionLayouts();

	initCommandPool();
	initCommandBuffer();
	initSyncObjects();
}

void VulkanApplication::mainLoop()
{
	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();
		//drawFrame();
		drawVolumeFrame();
	}

	device.waitIdle(); // wait for the queue(s) to become idle, finished executing the cmd?
}

void VulkanApplication::recordCommandBuffer(vk::CommandBuffer& commandBuffer, uint32_t imageIndex)
{
	commandBuffer.reset();
	const auto commandBufferBeginInfo = vk::CommandBufferBeginInfo{};
	commandBuffer.begin(commandBufferBeginInfo);
	const auto clearValues = std::vector<vk::ClearValue>{color::black}; // Color to clear imageIndex attachment in the framebuffer with
	const auto renderPassBeginInfo = vk::RenderPassBeginInfo{
		renderPass
		, framebuffers[imageIndex]
		, vk::Rect2D{{0, 0}, surfaceExtent}
		, clearValues
	};
	commandBuffer.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);
	commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, graphicPipeline);
	const auto bindingNumber = 0U;
	const auto offsets = std::vector<vk::DeviceSize>{0};
	// commandBuffer.bindIndexBuffer()
	commandBuffer.bindVertexBuffers(bindingNumber, vertexBuffer, offsets);
	commandBuffer.draw(3U, 1U, 0U, 0U);
	commandBuffer.endRenderPass();
	commandBuffer.end();
}
void VulkanApplication::drawFrame()
{
	// Get swapchain image
	std::ignore = device.waitForFences(isCommandBufferExecutedFence, VK_TRUE, UINT64_MAX);
	device.resetFences(isCommandBufferExecutedFence);
	const auto resultValue = device.acquireNextImageKHR(swapchain, UINT64_MAX, isAcquiredImageReadSemaphore, VK_NULL_HANDLE);

	// Record and submit commandbuffer for that image
	const auto imageIndex = resultValue.value;
	auto& commandBuffer = commandBuffers.front();
	recordCommandBuffer(commandBuffer, imageIndex);
	const auto stages = {vk::PipelineStageFlagBits::eColorAttachmentOutput};
	const auto waitStages = stages | std::ranges::to<std::vector<vk::PipelineStageFlags>>(); 
	auto submitInfo = vk::SubmitInfo{isAcquiredImageReadSemaphore, waitStages, commandBuffer, isImageRenderedSemaphore};
	queue.submit(submitInfo, isCommandBufferExecutedFence);

	// Present
	const auto presentInfo = vk::PresentInfoKHR{isImageRenderedSemaphore, swapchain, imageIndex};
	std::ignore = queue.presentKHR(presentInfo);
}

void VulkanApplication::cleanUp()
{
	device.destroy(isAcquiredImageReadSemaphore);
	device.destroy(isImageRenderedSemaphore);
	device.destroy(isCommandBufferExecutedFence);
	device.destroyCommandPool(commandPool);

	// ***** Volume rendering
	device.destroyEvent(raycastedEvent);
	device.destroyPipeline(computePipeline);
	device.destroyPipelineLayout(computePipelineLayout);
	device.destroyShaderModule(volumeShaderModule);
	device.destroyDescriptorSetLayout(descriptorSetLayout);
	device.destroySampler(sampler);
	device.destroyDescriptorPool(descriptorPool);
	device.freeMemory(stagingBufferMemory); device.destroyBuffer(stagingBuffer);
	device.destroyImageView(volumeImageView); device.destroyImageView(raycastedImageView);
	device.freeMemory(volumeImageMemory); device.destroyImage(volumeImage);
	device.freeMemory(raycastedImageMemory); device.destroyImage(raycastedImage);
	for (const vk::Framebuffer& framebuffer : framebuffers) device.destroyFramebuffer(framebuffer);
	for (const vk::ImageView& imageView : swapchainImageViews) device.destroyImageView(imageView);
	// ***** Volume rendering

	// Destroy the objects in reverse order of their creation order
	//for (const vk::Framebuffer& framebuffer : framebuffers) device.destroyFramebuffer(framebuffer);
	//device.destroyPipeline(graphicPipeline);
	//device.destroyPipelineLayout(pipelineLayout);
	//device.freeMemory(vertexBufferMemory);
	//device.destroyBuffer(vertexBuffer);
	//device.destroyRenderPass(renderPass);
	//for (const vk::ImageView& imageView : swapchainImageViews) device.destroyImageView(imageView);

	device.destroySwapchainKHR(swapchain);
	device.destroy();
	if (isValidationLayersEnabled) instance.destroyDebugUtilsMessengerEXT(debugMessenger);
	instance.destroySurfaceKHR(surface);
	instance.destroy();
	glfwDestroyWindow(window);
	glfwTerminate();
}

