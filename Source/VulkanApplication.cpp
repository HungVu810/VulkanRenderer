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
		const auto queueFamiliesIndex = std::views::iota(0U, queueFamiliesProperties.size());
		const auto isSuitable = [&](QueueFamilyIndex i)
		{
			const auto isGraphical = physicalDevice.getQueueFamilyProperties()[i].queueFlags & vk::QueueFlagBits::eGraphics;
			const auto isSurfaceSupported = physicalDevice.getSurfaceSupportKHR(i, surface); // For presentation support
			return isGraphical && isSurfaceSupported;
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

	// TODO: Volume data extraction, only needed by the shader?
	[[nodiscard]] inline auto getIntensity(std::span<Intensity, NUM_INTENSITY> slides, int z, int y, int x)
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
		assert(intensities.size() == NUM_INTENSITY); // Sanity check
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
	, isAcquiredImageRead{}
	, isImageRendered{}
	, isCommandBufferExecuted{}
{
}

VulkanApplication::~VulkanApplication() noexcept {}

void VulkanApplication::run() noexcept
{
	try
	{
		validateShadersWorker = std::thread{validateShaders, std::ref(shaderMap)};
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
		, imageCount
		, surfaceFormat.format
		, surfaceFormat.colorSpace
		, surfaceExtent
		, 1 // Number of layers per image, more than 1 for stereoscopic application
		, vk::ImageUsageFlagBits::eColorAttachment // render directly onto the framebuffers (no post-processing)
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
	validateShadersWorker.join();

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
	vertexBufferMemory = allocateMemory(device, physicalDevice, memoryRequirements);
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

void VulkanApplication::initComputePipeline()
{
	importVolumeDataWorker.join();

	const auto volumeShaderBinaryData = getShaderBinaryData(shaderMap, "VolumeRendering.comp");
	const auto volumeShaderModuleCreateInfo = vk::ShaderModuleCreateInfo{{}, volumeShaderBinaryData};
	const auto volumeShaderModule = device.createShaderModule(volumeShaderModuleCreateInfo);
	const auto computeShaderStageCreateInfo = vk::PipelineShaderStageCreateInfo{
		{}
		, vk::ShaderStageFlagBits::eCompute
		, volumeShaderModule
		, "main"
	};

	const auto raycastedImageCreateInfo = vk::ImageCreateInfo{
		{}
		, vk::ImageType::e2D
		, surfaceFormat.format
		, vk::Extent3D{surfaceExtent.width, surfaceExtent.height, 1}
		, 1 // The only mip level for this image
		, 1 // Single layer, no stereo-rendering
		, vk::SampleCountFlagBits::e1 // One sample, no multisampling
		, vk::ImageTiling::eOptimal
		, vk::ImageUsageFlagBits::eColorAttachment
		, vk::SharingMode::eExclusive
		, getSuitableQueueFamilies(physicalDevice, surface).front().first
	};
	const auto raycastedImage = device.createImage(raycastedImageCreateInfo);
	const auto raycastedImageMemoryRequirements = device.getImageMemoryRequirements(raycastedImage);
	const auto raycastedImageMemory = allocateMemory(device, physicalDevice, raycastedImageMemoryRequirements);
	device.bindImageMemory(raycastedImage, raycastedImageMemory, 0);
	const auto raycastedImageViewCreateInfo = vk::ImageViewCreateInfo{
		{}
		, raycastedImage
		, vk::ImageViewType::e2D
		, surfaceFormat.format
	};
	const auto raycastedImageView = device.createImageView(raycastedImageViewCreateInfo);
	// TODO: attach this imageview to the framebuffer, and renderpass so we can modifies it in the compute shader
	// TODO: or make it available via descriptor set?
	// TODO; the 3D volumn data has to be inn a descriptor set

	// const auto queueFamilies = getSuitableQueueFamilies(physicalDevice, surface);
	// createBuffer(device, intensities, queueFamilies, vk::BufferUsageFlagBits::)

	// storage image descriptor (for the image to be ray casted)
	// 3D texture sampled descriptor (for the 3D volume data)
	// run the compute shader in parallel each pixel, instead of going through each of them one by one

	//// TODO: upload the 3D volume texture data via the descriptor set
	//// TODO: generate image grid?
	//// TODO: Bound the descripor sets when record the command buffer

	// TODO: how to present the image in the compute shader after finished ray castingn?

	//const auto numDescriptorSet = 1;
	//const auto descriptorPoolSize = vk::DescriptorPoolSize{vk::DescriptorType::};
	//vk::DescriptorPoolCreateInfo{vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, }
	//// device.createDescriptorPool()
	//// vk::DescriptorSetAllocateInfo{}
	//// device.allocateDescriptorSets()

	//const auto descriptorSetLayoutBinding = vk::DescriptorSetLayoutBinding{
	//	0
	//	, vk::DescriptorType::eStorageImage
	//	// , vk::ShaderStageFlagBits::eCompute
	//};
	//const auto descriptorSetLayoutCreateInfo = vk::DescriptorSetLayoutCreateInfo{{}, descriptorSetLayoutBinding};
	//const auto descriptorSetLayout = device.createDescriptorSetLayout(descriptorSetLayoutCreateInfo);

	//const auto pipelineLayoutCreateInfo = vk::PipelineLayoutCreateInfo{{}, descriptorSetLayout};
	//const auto pipelineLayout = device.createPipelineLayout(pipelineLayoutCreateInfo);

	////const auto computePipelineCreateInfo = vk::ComputePipelineCreateInfo{};
	////const auto computePipeline = device.createComputePipeline({}, computePipelineCreateInfo);

	//device.destroyPipeline();
	device.destroyImageView(raycastedImageView);
	device.destroyImage(raycastedImage);
	device.destroyDescriptorSetLayout(); // but this in the cleanUp();
	device.destroyShaderModule(volumeShaderModule);
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
	isAcquiredImageRead = device.createSemaphore(semCreateInfo);
	isImageRendered = device.createSemaphore(semCreateInfo);
	isCommandBufferExecuted = device.createFence(fenceCreateInfo);
}

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
	initImageViews();
	initRenderPass();

	initGraphicPipeline();
	initComputePipeline();

	initFrameBuffer();
	initCommandPool();
	initCommandBuffer();
	initSyncObjects();
}

void VulkanApplication::mainLoop()
{
	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();
		drawFrame();
	}

	device.waitIdle();
}

void VulkanApplication::recordCommandBuffer(vk::CommandBuffer& commandBuffer, uint32_t imageIndex)
{
	commandBuffer.reset();
	const auto commandBufferBeginInfo = vk::CommandBufferBeginInfo{};
	commandBuffer.begin(commandBufferBeginInfo);
	const auto clearColorValue = vk::ClearColorValue{0.0f, 0.0f, 0.0f, 1.0f}; // Color to clear the screen with
	const auto clearValues = std::vector<vk::ClearValue>{clearColorValue};
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
	commandBuffer.bindVertexBuffers(bindingNumber, vertexBuffer, offsets);
	commandBuffer.draw(3U, 1U, 0U, 0U);
	commandBuffer.endRenderPass();
	commandBuffer.end();
}
void VulkanApplication::drawFrame()
{
	// Get swapchain image
	std::ignore = device.waitForFences(isCommandBufferExecuted, VK_TRUE, UINT64_MAX);
	device.resetFences(isCommandBufferExecuted);
	const auto resultValue = device.acquireNextImageKHR(swapchain, UINT64_MAX, isAcquiredImageRead, VK_NULL_HANDLE);

	// Record and submit commandbuffer for that image
	const auto imageIndex = resultValue.value;
	auto& commandBuffer = commandBuffers.front();
	recordCommandBuffer(commandBuffer, imageIndex);
	const auto stages = {vk::PipelineStageFlagBits::eColorAttachmentOutput};
	const auto waitStages = stages | std::ranges::to<std::vector<vk::PipelineStageFlags>>(); 
	auto submitInfo = vk::SubmitInfo{isAcquiredImageRead, waitStages, commandBuffer, isImageRendered};
	queue.submit(submitInfo, isCommandBufferExecuted);

	// Present
	const auto presentInfo = vk::PresentInfoKHR{isImageRendered, swapchain, imageIndex};
	std::ignore = queue.presentKHR(presentInfo);
}

void VulkanApplication::cleanUp()
{
	// Destroy the objects in reverse order of their creation order
	device.destroy(isAcquiredImageRead);
	device.destroy(isImageRendered);
	device.destroy(isCommandBufferExecuted);
	device.destroyCommandPool(commandPool);
	for (const vk::Framebuffer& framebuffer : framebuffers) device.destroyFramebuffer(framebuffer);
	device.destroyPipeline(graphicPipeline);
	device.destroyPipelineLayout(pipelineLayout);
	device.freeMemory(vertexBufferMemory);
	device.destroyBuffer(vertexBuffer);
	device.destroyRenderPass(renderPass);
	for (const vk::ImageView& imageView : swapchainImageViews) device.destroyImageView(imageView);
	device.destroySwapchainKHR(swapchain);
	device.destroy();
	if (isValidationLayersEnabled) instance.destroyDebugUtilsMessengerEXT(debugMessenger);
	instance.destroySurfaceKHR(surface);
	instance.destroy();
	glfwDestroyWindow(window);
	glfwTerminate();
}


