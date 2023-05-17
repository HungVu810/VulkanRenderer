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
			return surfaceFormat.format == toVulkanFormat<format::Image>()
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
	// Imgui
	, imguiRenderPass{}
	, imguiSwapchainImageViews{}
	, imguiDescriptorPool{}
	, imguiFramebuffers{}
	, imguiCommandPool{}
	, imguiCommandBuffers{}
{
}

VulkanApplication::~VulkanApplication() noexcept {}

void VulkanApplication::run(const RunInfo& runInfo) noexcept // All exceptions are handled in this function so we can clean up the resources thereafter.
{
	try
	{
		initWindow(runInfo.windowName); // Must be before initVulkan()
		initVulkan(runInfo);
		// Application info only valid after initVulkan
		const auto applicationInfo = ApplicationInfo{
			window
			, instance
			, surface
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
		renderLoop(runInfo.renderFrame, applicationInfo, runInfo.windowName);
		runInfo.postRenderLoop(applicationInfo);
		cleanUp(); // Can't put in the class' destructor due to potential exceptions
	}
	catch (const std::exception& except)
	{
		std::cerr << tag::exception << except.what() << std::endl;
	}
}

void VulkanApplication::initWindow(std::string_view windowName = "MyWindow")
{
	if (glfwInit() != GLFW_TRUE) throw std::runtime_error{"Failed to initalize GLFW"};
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // Do not create an OpenGL context
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
	window = glfwCreateWindow(WIDTH, HEIGHT, windowName.data(), nullptr, nullptr);
	if (!window) throw std::runtime_error{"Can't create window"};
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
	//initSwapChain(runInfo.swapchainImageUsage);
	initSwapChain();
	initCommandPool();
	initCommandBuffer();
	initSyncObjects();
	initImGui();
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

//void VulkanApplication::initSwapChain(vk::ImageUsageFlagBits swapchainImageUsage)
void VulkanApplication::initSwapChain()
{
	const auto surfaceAttributes = getSurfaceAttributes(surface, physicalDevice);
	surfaceFormat = getSuitableSurfaceFormat(surfaceAttributes.formats);
	surfaceExtent = getSuitableSurfaceExtent(window, surfaceAttributes.capabilities);
	const auto presentMode = getSuitablePresentMode(surfaceAttributes.presentModes);
	auto imageCount = surfaceAttributes.capabilities.minImageCount + 1;
	if (surfaceAttributes.capabilities.maxImageCount != 0) // Not infinite
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
		, vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst // Expected image usages to prevent from transitioning layout into conflicting types
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

//TODO: Check if the runInfo already provdie a renderpass, grpahic pipeline, framebuffer,...?

void VulkanApplication::initImGui() // https://github.com/ocornut/imgui/blob/master/examples/example_glfw_vulkan/main.cpp
{
	// Imgui is layered onto top of our window
	// See example_glfw_vulkan and imgui_impl_vulkan.cpp

// #ifdef USE_IMGUI && USE_IMGUI == 1
	// Pass the imguiWindow this to the applicatoininfo
	// application only setup begin/end of window, the vulkanapplication manage frame inddex imgui
	// All the ImGui_ImplVulkanH_XXX structures/functions are optional helpers used by the demo.

	//IM_ASSERT(wd->Frames == nullptr);
	//wd->Frames = (ImGui_ImplVulkanH_Frame*)IM_ALLOC(sizeof(ImGui_ImplVulkanH_Frame) * wd->ImageCount);
	//wd->FrameSemaphores = (ImGui_ImplVulkanH_FrameSemaphores*)IM_ALLOC(sizeof(ImGui_ImplVulkanH_FrameSemaphores) * wd->ImageCount);
	//memset(wd->Frames, 0, sizeof(wd->Frames[0]) * wd->ImageCount);
	//memset(wd->FrameSemaphores, 0, sizeof(wd->FrameSemaphores[0]) * wd->ImageCount);
	//for (uint32_t i = 0; i < wd->ImageCount; i++)
	//    wd->Frames[i].Backbuffer = backbuffers[i];

	// Reserved for ImGui
	initImGuiDescriptorPool();
	initImGuiRenderPass();
	initImGuiImageViews();
	initImGuiFrameBuffer();
	initImGuiCommandPool();
	initImGuiCommandBuffer();

	// Setup Dear ImGui context
	IMGUI_CHECKVERSION();
	const auto imguiContext = ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	//io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
	//io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
	//TODO: variable Toggle imgui log messages

	// Setup Dear ImGui style
	ImGui::StyleColorsDark();
	//ImGui::StyleColorsLight();

	const auto surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(surface);
	// Setup Platform/Renderer backends
	assertm(surfaceCapabilities.minImageCount >= 2, "Min image count for swapchain is not greater than or equal to 2");
	ImGui_ImplGlfw_InitForVulkan(window, true);
	ImGui_ImplVulkan_InitInfo init_info = {}; // Used to create an imgui pipeline, done by ImGui_ImplVulkan_Init because imgui need to bring in its own vertex/fragment shader modules for the UI
	init_info.Instance = instance;
	init_info.PhysicalDevice = physicalDevice;
	init_info.Device = device;
	init_info.QueueFamily = getSuitableQueueFamilies(physicalDevice, surface).front().first;
	init_info.Queue = queue;
	init_info.PipelineCache = VK_NULL_HANDLE;
	init_info.DescriptorPool = imguiDescriptorPool;
	init_info.Subpass = 0; // The index of the subpass in the imgui renderpass that will use the created imgui pipeline
	init_info.MinImageCount = surfaceCapabilities.minImageCount; // >= 2
	init_info.ImageCount = surfaceCapabilities.maxImageCount; // >= MinImageCount
	init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT; // >= VK_SAMPLE_COUNT_1_BIT (0 -> default to VK_SAMPLE_COUNT_1_BIT)
	init_info.Allocator = nullptr; // vk::AllocationCallbacks used in all of vk::Function
	init_info.CheckVkResultFn = checkVkResult;
	ImGui_ImplVulkan_Init(&init_info, imguiRenderPass);

	// Upload imgui font
	submitCommandBufferOnceSynced(device, queue, commandBuffers.front(), [](const vk::CommandBuffer& commandBuffer){
			ImGui_ImplVulkan_CreateFontsTexture(commandBuffer);
	});
	ImGui_ImplVulkan_DestroyFontUploadObjects();

// #endif
}

void VulkanApplication::initImGuiDescriptorPool()
{
	const auto descriptorPoolSizes = std::vector<vk::DescriptorPoolSize>{
		{vk::DescriptorType::eSampler, 1000}
		, {vk::DescriptorType::eCombinedImageSampler, 1000}
		, {vk::DescriptorType::eSampledImage, 1000}
		, {vk::DescriptorType::eStorageImage, 1000}
		, {vk::DescriptorType::eUniformTexelBuffer, 1000}
		, {vk::DescriptorType::eStorageTexelBuffer, 1000}
		, {vk::DescriptorType::eUniformBuffer, 1000}
		, {vk::DescriptorType::eStorageBuffer, 1000}
		, {vk::DescriptorType::eUniformBufferDynamic, 1000}
		, {vk::DescriptorType::eStorageBufferDynamic, 1000}
		, {vk::DescriptorType::eInputAttachment, 1000}
	};
	imguiDescriptorPool = device.createDescriptorPool(vk::DescriptorPoolCreateInfo{
		vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet
		, 1000
		, descriptorPoolSizes
	});
}
void VulkanApplication::initImGuiImageViews()
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
	imguiSwapchainImageViews = images | std::views::transform(toImageView) | std::ranges::to<std::vector>();
}
// Load the image from the previous render command buffer, unless the render command buffer using the same render pass then we can clear it
//********** This is expected from the previous command buffer but is undefined if we use the same renderpass
void VulkanApplication::initImGuiRenderPass()
{
	const auto attachmentDescription = vk::AttachmentDescription{
		{}
		, surfaceFormat.format
		, vk::SampleCountFlagBits::e1 // One sample, no multisampling
		, vk::AttachmentLoadOp::eLoad // Load the image from the previous render command buffer, unless the render command buffer using the same render pass then we can clear it
		, vk::AttachmentStoreOp::eStore
		, vk::AttachmentLoadOp::eDontCare
		, vk::AttachmentStoreOp::eDontCare
		, vk::ImageLayout::eColorAttachmentOptimal // Expected layout at the beginning of the renderpass, ********** This is expected from the previous command buffer but is undefined if we use the same renderpass
		, vk::ImageLayout::ePresentSrcKHR // Expected layout at the end of the renderpass, final layout for presentation
	};
	const auto attachmentReference = vk::AttachmentReference{
		0U // In the array of attachmentDescriptions
		, vk::ImageLayout::eColorAttachmentOptimal // Expected layout at the end of this subpass, will be automatically transitioned by the graphic subpass
	};
	const auto subpassDescription = vk::SubpassDescription{ // The index of this subpass among the subpass description is specified GraphicsPipelineCreateInfo
		{}
		, vk::PipelineBindPoint::eGraphics
		, {}
		, attachmentReference
	};
	const auto subpassDependency = vk::SubpassDependency{
		VK_SUBPASS_EXTERNAL // The one before
		, 0U // The current one
		, vk::PipelineStageFlagBits::eColorAttachmentOutput
		, vk::PipelineStageFlagBits::eColorAttachmentOutput
		, vk::AccessFlagBits::eNone
		, vk::AccessFlagBits::eColorAttachmentWrite
	};
	imguiRenderPass = device.createRenderPass(vk::RenderPassCreateInfo{
		{}
		, attachmentDescription
		, subpassDescription
		, subpassDependency
	});
}
void VulkanApplication::initImGuiFrameBuffer()
{
	const auto toFramebuffer = [&](const vk::ImageView& imageView)
	{
		const auto framebufferCreateInfo = vk::FramebufferCreateInfo{
			{}
			, imguiRenderPass
			, imageView // Attachments
			, surfaceExtent.width
			, surfaceExtent.height
			, 1 // Single layer, not doing stereo-rendering
		};
		return device.createFramebuffer(framebufferCreateInfo);
	};
	imguiFramebuffers = imguiSwapchainImageViews | std::views::transform(toFramebuffer) | std::ranges::to<std::vector>();
}
void VulkanApplication::initImGuiCommandPool()
{
	const auto queueFamilies = getSuitableQueueFamilies(physicalDevice, surface);
	const auto commandPoolCreateInfo = vk::CommandPoolCreateInfo{
		vk::CommandPoolCreateFlagBits::eResetCommandBuffer // reset and rerecord command buffer
		, queueFamilies.front().first
	};
	imguiCommandPool = device.createCommandPool(commandPoolCreateInfo);
}
void VulkanApplication::initImGuiCommandBuffer()
{
	const auto allocateInfo = vk::CommandBufferAllocateInfo{
		imguiCommandPool
		, vk::CommandBufferLevel::ePrimary // TODO: Secondary for UI? so we don't have to sync between 2 primary? or do we still need to sync between a primary and a secondary?
		, 1
	};
	imguiCommandBuffers = device.allocateCommandBuffers(allocateInfo);
}

void VulkanApplication::renderLoop(const RenderFrameFunction& renderFrame, const ApplicationInfo& applicationInfo, std::string_view windowName = "MyWindow") // Always render in the order from the back to the front with respect to the viewport depth
{
	// Must have an event to prevent data race between the imgui command buffer and the application rendr ocmmand buufer if they read/write to a descriptor
	const auto renderCommandBufferEvent = device.createEvent(vk::EventCreateInfo{vk::EventCreateFlagBits::eDeviceOnly});

	// TODO: Rebuild the swapchain if resizing
	// TODO: Inflight frames

	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents(); // Pass this is imgui, see opengl implementation

		const auto preFrameRender = glfwGetTime();

		ImGui_ImplVulkan_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		const auto isFirstFrame = device.getFenceStatus(isCommandBufferExecutedFence) == vk::Result::eSuccess;
		std::ignore = device.waitForFences(isCommandBufferExecutedFence, VK_TRUE, std::numeric_limits<uint64_t>::max()); // Avoid modifying the command buffer when it's in used by the device
		device.resetFences(isCommandBufferExecutedFence);
		const auto resultValue = device.acquireNextImageKHR(swapchain, std::numeric_limits<uint64_t>::max(), isAcquiredImageReadSemaphore); // Semaphore will be raised when the acquired image is finished reading by the engine
		if (resultValue.result != vk::Result::eSuccess) throw std::runtime_error{"Failed to acquire the next image index."};
		const auto imageIndex = resultValue.value;

		///////////////////////
		// TODO: keep vk::ImageUsageFlagBits::eColorAttachment as the default for swapchain images, make sure the application transfer back to this attachment in order for it to be used by imgui
		///////////////////////
		renderFrame(applicationInfo, imageIndex, isFirstFrame); // renderrFrame = a function with command to be exeuted only?
		const auto commandBuffer = commandBuffers.front(); // rendercomandbuffer
		commandBuffer.setEvent(renderCommandBufferEvent, vk::PipelineStageFlagBits::eComputeShader);
		commandBuffer.end();

		// Imgui is likely to share resource with the applicationss' commandbuffers commands, sync this imguiCommandbuffer and the application render commandbuffer

		ImGui::ShowDemoWindow();
		//// For each imgui window
		//ImGui::Begin("Hello, world!");
		//ImGui::End();
        ImGui::Render(); // Get the user inputs from imgui first then we will construct a frame with them
        ImDrawData* draw_data = ImGui::GetDrawData();
		//FrameRender(wd, draw_data); ------------------------------------------
		device.resetCommandPool(imguiCommandPool);
		const auto imguiCommandBuffer = imguiCommandBuffers.front();
		imguiCommandBuffer.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
		imguiCommandBuffer.waitEvents(renderCommandBufferEvent, vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTopOfPipe, {}, {}, {});
		const auto clearval = static_cast<vk::ClearValue>(color::black);
		imguiCommandBuffer.beginRenderPass(vk::RenderPassBeginInfo{
				imguiRenderPass
				, imguiFramebuffers[imageIndex]
				, vk::Rect2D{vk::Offset2D{0, 0}, vk::Extent2D{surfaceExtent.width, surfaceExtent.height}}
				, clearval
			}, vk::SubpassContents::eInline);
		ImGui_ImplVulkan_RenderDrawData(draw_data, imguiCommandBuffer); // Record imgui primitives into command buffer via imgui pipeline
		imguiCommandBuffer.endRenderPass();
		imguiCommandBuffer.end();

		const auto waitStages = std::vector<vk::PipelineStageFlags>{
			//vk::PipelineStageFlagBits::eComputeShader // Don't need this if we have the imgui renderpass after the compute shader
			 vk::PipelineStageFlagBits::eColorAttachmentOutput // Imgui, doesn't need to be a seperate wait stage if the main app using renderpass
		};
		const auto commbufs = {commandBuffer, imguiCommandBuffer};
		//const auto commbufs = {commandBuffer};
		queue.submit(vk::SubmitInfo{
			isAcquiredImageReadSemaphore // Wait for the image to be finished reading, then we will modify it via the commands in the commandBuffers
			, waitStages 
			, commbufs
			, isImageRenderedSemaphore // Raise when finished executing the commands
		}, isCommandBufferExecutedFence); // Raise when finished executing the commands

		//FramePresent(wd); ------------------------------------
		const auto presentResult = queue.presentKHR(vk::PresentInfoKHR{
			isImageRenderedSemaphore
			, swapchain
			, imageIndex
		});
		if (presentResult != vk::Result::eSuccess) throw std::runtime_error{"Failed to present image."};

		const auto postFrameRender = glfwGetTime();
		const auto frameRenderTime = postFrameRender - preFrameRender;
		const auto framesPerSecond = static_cast<int>(std::round(1 / frameRenderTime));
		glfwSetWindowTitle(window, (std::string{windowName} + " - FPS: " + std::to_string(framesPerSecond)).data());
	}
	device.waitIdle(); // Wait for all the fences to be unsignaled before clean up

	device.destroyEvent(renderCommandBufferEvent);
	device.freeCommandBuffers(imguiCommandPool, imguiCommandBuffers);
	device.destroyCommandPool(imguiCommandPool);
	for (const auto framebuffer : imguiFramebuffers)
	{
		device.destroyFramebuffer(framebuffer);
	}
	device.destroyDescriptorPool(imguiDescriptorPool);
	for (const auto imageview : imguiSwapchainImageViews)
	{
		device.destroyImageView(imageview);
	}
	device.destroyRenderPass(imguiRenderPass);
}

void VulkanApplication::renderImGui()
{
}

void VulkanApplication::cleanUp() // Destroy the objects in reverse order of their creation order
{
	ImGui_ImplVulkan_Shutdown();
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


