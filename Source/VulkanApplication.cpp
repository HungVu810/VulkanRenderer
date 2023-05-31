#include "VulkanApplication.h" // Do not place this right above the VULKAN_HPP_DEFAULT macro
#include "Utilities.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"
#include <ranges>
#include <algorithm>
#include <stdexcept>
#include <cstdint> // Needed for uint32_t
#include <limits> // Needed for std::numeric_limits
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
		const auto isSupported = std::ranges::find(layersProperties, validationLayerName, [](const auto& layerProperties){return std::string_view(layerProperties.layerName);}) != std::end(layersProperties);
		if (!isSupported) throw std::runtime_error{"Requested validation layers name can't be found."};
		strings.push_back(validationLayerName);
		return strings;
	}
	[[nodiscard]] auto getEnabledInstanceExtensionsName()
	{
		auto extensionCount = uint32_t{0};
		const auto glfwExtensionsRawName = glfwGetRequiredInstanceExtensions(&extensionCount);
		if (!glfwExtensionsRawName) throw std::runtime_error{"Vulkan's surface extensions for window creation can not be found"};
		auto extensionsName = std::vector<std::string>{glfwExtensionsRawName, glfwExtensionsRawName + extensionCount};
		if (isValidationLayersEnabled) extensionsName.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
		return extensionsName;
	}
	[[nodiscard]] auto getDeviceExtensionsName() noexcept
	{
		return std::vector<std::string>{VK_KHR_SWAPCHAIN_EXTENSION_NAME};
	}
	[[nodiscard]] auto getSurfaceAttributes(const vk::SurfaceKHR& surface, const vk::PhysicalDevice& physicalDevice) noexcept
	{
		return SurfaceAttributes{
			physicalDevice.getSurfaceCapabilitiesKHR(surface),
			physicalDevice.getSurfaceFormatsKHR(surface),
			physicalDevice.getSurfacePresentModesKHR(surface)
		};
	}
	[[nodiscard]] auto getSuitablePhysicalDevice(const std::vector<vk::PhysicalDevice>& physicalDevices, const vk::SurfaceKHR& surface)
	{
		const auto deviceExtensionsName = getDeviceExtensionsName();
		auto suitablePhysicalDevices = physicalDevices | std::views::filter([&](const vk::PhysicalDevice& physicalDevice)
		{
			const auto surfaceAttributes = getSurfaceAttributes(surface, physicalDevice);
			const auto deviceProperties = physicalDevice.getProperties();
			const auto deviceFeatures = physicalDevice.getFeatures();
			const auto supportedDeviceExtensionsName = physicalDevice.enumerateDeviceExtensionProperties() | std::views::transform([](const vk::ExtensionProperties& properties)
			{
				return static_cast<std::string_view>(properties.extensionName);
			});
			return deviceFeatures.geometryShader
				&& !surfaceAttributes.formats.empty()
				&& !surfaceAttributes.presentModes.empty()
				&& std::ranges::all_of(deviceExtensionsName, [&](const std::string_view& deviceExtensionName)
				{
					return std::ranges::find(supportedDeviceExtensionsName, deviceExtensionName) != supportedDeviceExtensionsName.end();
				});
		});
		if (suitablePhysicalDevices.empty()) throw std::runtime_error{ "Can't find a suitable GPU." };
		const auto discretePhysicalDeviceIter = std::ranges::find_if(suitablePhysicalDevices, [](const vk::PhysicalDevice& physicalDevice)
		{
			return physicalDevice.getProperties().deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
		});
		if (discretePhysicalDeviceIter != suitablePhysicalDevices.end()) return *discretePhysicalDeviceIter;
		else
		{
			std::cout << tag::warning << "Using a non-discrete device for rendering.\n";
			return suitablePhysicalDevices.front(); // Just using a random integrated device
		}
	}
	[[nodiscard]] auto getSuitableSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& surfaceFormats) noexcept
	{
		const auto surfaceFormatIter = std::ranges::find_if(surfaceFormats, [](const vk::SurfaceFormatKHR& surfaceFormat)
		{
			return surfaceFormat.format == toVulkanFormat<format::Image>()
				&& surfaceFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear;
		});
		if (surfaceFormatIter != surfaceFormats.end()) return *surfaceFormatIter; 
		else return surfaceFormats.front();
	}
	[[nodiscard]] auto getSuitablePresentMode(const std::vector<vk::PresentModeKHR>& presentModes) noexcept
	{
		const auto presentModesIter = std::ranges::find_if(presentModes, [](const vk::PresentModeKHR& presentMode)
		{
			return presentMode == vk::PresentModeKHR::eMailbox;
		});
		if (presentModesIter != presentModes.end()) return *presentModesIter;
		else return vk::PresentModeKHR::eFifo;
	}
	[[nodiscard]] auto getSuitableSurfaceExtent(GLFWwindow* window, const vk::SurfaceCapabilitiesKHR& surfaceCapabilities) noexcept
	{
		if (surfaceCapabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) return surfaceCapabilities.currentExtent;
		else
		{
			auto width = int{0};
			auto height = int{0};
			glfwGetFramebufferSize(window, &width, &height);
			const auto suitableWidth = std::clamp(static_cast<uint32_t>(width), surfaceCapabilities.minImageExtent.width, surfaceCapabilities.maxImageExtent.width);
			const auto suitableHeight = std::clamp(static_cast<uint32_t>(height), surfaceCapabilities.minImageExtent.height, surfaceCapabilities.maxImageExtent.height);
			const auto suitableExtent = vk::Extent2D{suitableWidth, suitableHeight};
			return suitableExtent;
		}
	}
	[[nodiscard]] auto getSuitableQueueFamilies(const vk::PhysicalDevice& physicalDevice, const vk::SurfaceKHR& surface) noexcept
	{
		const auto queueFamiliesProperties = physicalDevice.getQueueFamilyProperties();
		auto queueFamilies = queueFamiliesProperties
			| std::views::enumerate
			| std::views::filter([&](const auto& indexedQueueFamilyProperties)
			{
				const auto& [queueFamilyIndex, queueFamilyProperties] = indexedQueueFamilyProperties;
				const auto isGraphical = queueFamilyProperties.queueFlags & vk::QueueFlagBits::eGraphics;
				const auto isCompute = queueFamilyProperties.queueFlags & vk::QueueFlagBits::eCompute;
				const auto isSurfaceSupported = physicalDevice.getSurfaceSupportKHR(queueFamilyIndex, surface); // For presentation support
				return isGraphical && isCompute && isSurfaceSupported;
			})
			| std::views::transform([](const auto& indexedQueueFamilyProperties)
			{
				const auto& [queueFamilyIndex, queueFamilyProperties] = indexedQueueFamilyProperties;
				return QueueFamily{static_cast<QueueFamilyIndex>(queueFamilyIndex), {1.0f}}; // One queue of priority level 1
			})
			| std::ranges::to<std::vector>();
		return queueFamilies;
	}
}

VulkanApplication::VulkanApplication()
	: applicationInfo{std::nullopt}
	, window{nullptr}
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
		initWindow(runInfo); // Must be before initVulkan()
		initVulkan();
		applicationInfo = ApplicationInfo{
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
			, commandBuffers.front()
		};
		// Application info only valid after initVulkan
		runInfo.preRenderLoop(applicationInfo.value());
		renderLoop(runInfo, applicationInfo.value());
		runInfo.postRenderLoop(applicationInfo.value());
		cleanUp(); // Can't put in the destructor due to potential exceptions
	}
	catch (const std::exception& except)
	{
		std::cerr << tag::exception << except.what() << std::endl;
	}
}

inline const ApplicationInfo& VulkanApplication::getApplicationInfo() const noexcept
{
	assertm(applicationInfo.has_value(), "Forgot to call run() for the current VulkanApplication");
	return applicationInfo.value();
}

void VulkanApplication::initWindow(const RunInfo& runInfo)
{
	if (glfwInit() != GLFW_TRUE) throw std::runtime_error{"Failed to initialize GLFW"};
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // Do not create an OpenGL context
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
	window = glfwCreateWindow(WIDTH, HEIGHT, runInfo.windowName.data(), nullptr, nullptr);
	if (!window) throw std::runtime_error{"Can't create window"};
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
	instance = vk::createInstance(vk::InstanceCreateInfo{
		{},
		&applicationInfo,
		enabledLayersNameProxy,
		enabledInstanceExtensionsNameProxy,
		isValidationLayersEnabled ? &debugMessengerCreateInfo : nullptr // Extend create info with debug messenger
	});
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
	const auto queueCreateInfos = queueFamilies
		| std::views::transform([](const QueueFamily& queueFamily){return vk::DeviceQueueCreateInfo{{}, queueFamily.first, queueFamily.second};})
		| std::ranges::to<std::vector>();
	device = physicalDevice.createDevice(vk::DeviceCreateInfo{
		{},
		queueCreateInfos,
		enabledLayersNameProxy,
		enabledDeviceExtensionsNameProxy,
		&physicalDeviceFeatures
	});
}
void VulkanApplication::initQueue()
{
	const auto queueFamilyIndex = 0; // TODO: Always pick the first queueFamilyIndex for now
	const auto queueIndex = 0;       // TODO: Always pick the first queue of type queueFamilyIndex for now
	queue = device.getQueue(queueFamilyIndex, queueIndex);
}
void VulkanApplication::initSwapChain()
{
	const auto surfaceAttributes = getSurfaceAttributes(surface, physicalDevice);
	surfaceFormat = getSuitableSurfaceFormat(surfaceAttributes.formats);
	checkFormatFeatures(physicalDevice, surfaceFormat.format, vk::FormatFeatureFlagBits::eColorAttachment);
	checkFormatFeatures(physicalDevice, surfaceFormat.format, vk::FormatFeatureFlagBits::eTransferDst); // The application could use a compute shader to render to a target image and transfer the data over a swapchain image
	surfaceExtent = getSuitableSurfaceExtent(window, surfaceAttributes.capabilities);
	auto imageCount = surfaceAttributes.capabilities.minImageCount + 1;
	if (surfaceAttributes.capabilities.maxImageCount != 0) // Not infinite
	{
		imageCount = std::min(imageCount, surfaceAttributes.capabilities.maxImageCount);
	}
	swapchain = device.createSwapchainKHR(vk::SwapchainCreateInfoKHR{
		{}
		, surface
		, imageCount
		, surfaceFormat.format
		, surfaceFormat.colorSpace
		, surfaceExtent
		, 1 // Single layer, no stereo-rendering
		, vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst // Expected image usages to prevent from transitioning layout into conflicting types
		, vk::SharingMode::eExclusive // Drawing and presentation are done with one physical device's queue
		, getSuitableQueueFamilies(physicalDevice, surface).front().first
		, surfaceAttributes.capabilities.currentTransform
		, vk::CompositeAlphaFlagBitsKHR::eOpaque
		, getSuitablePresentMode(surfaceAttributes.presentModes)
		, VK_TRUE
		, VK_NULL_HANDLE
	});
}
void VulkanApplication::initCommandPool()
{
	commandPool = device.createCommandPool(vk::CommandPoolCreateInfo{
		vk::CommandPoolCreateFlagBits::eResetCommandBuffer // Reset and rerecord command buffer
		, getSuitableQueueFamilies(physicalDevice, surface).front().first
	});
}
void VulkanApplication::initCommandBuffer()
{
	commandBuffers = device.allocateCommandBuffers(vk::CommandBufferAllocateInfo{
		commandPool
		, vk::CommandBufferLevel::ePrimary
		, 1
	});
}
void VulkanApplication::initSyncObjects()
{
	isAcquiredImageReadSemaphore = device.createSemaphore(vk::SemaphoreCreateInfo{});
	isImageRenderedSemaphore = device.createSemaphore(vk::SemaphoreCreateInfo{});
	isCommandBufferExecutedFence = device.createFence(vk::FenceCreateInfo{vk::FenceCreateFlagBits::eSignaled}); // First frame doesn't have to wait for the unexisted previous image
}

void VulkanApplication::initImGui()
{
	// https://github.com/ocornut/imgui/wiki
	// https://github.com/epezent/implot
	// https://github.com/CedricGuillemet/ImGuizmo

	// Reserved for imgui
	initImGuiDescriptorPool();
	initImGuiRenderPass();
	initImGuiImageViews();
	initImGuiFrameBuffer();
	initImGuiCommandPool();
	initImGuiCommandBuffer();

	// Setup Dear ImGui context
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();

	// Setup Platform/Renderer backends
	const auto surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(surface);
	assertm(surfaceCapabilities.minImageCount >= 2, "Min image count for the swapchain is not greater than or equal to 2");
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
		return device.createImageView(vk::ImageViewCreateInfo{
			{}
			, image
			, vk::ImageViewType::e2D
			, surfaceFormat.format
			, {vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity}
			, {vk::ImageAspectFlagBits::eColor, 0u, 1u, 0u, 1u}
		});
	};
	imguiSwapchainImageViews = images | std::views::transform(toImageView) | std::ranges::to<std::vector>();
}
void VulkanApplication::initImGuiRenderPass()
{
	const auto attachmentDescription = vk::AttachmentDescription{
		{}
		, surfaceFormat.format
		, vk::SampleCountFlagBits::e1 // One sample, no multisampling
		, vk::AttachmentLoadOp::eLoad // Load the swapchain image rendered by the application
		, vk::AttachmentStoreOp::eStore
		, vk::AttachmentLoadOp::eDontCare
		, vk::AttachmentStoreOp::eDontCare
		, vk::ImageLayout::eColorAttachmentOptimal // Expected layout at the beginning of the renderpass
		, vk::ImageLayout::ePresentSrcKHR // Expected layout at the end of the renderpass, final layout for presentation
	};
	const auto attachmentReference = vk::AttachmentReference{
		0U // In the array of attachmentDescriptions
		, vk::ImageLayout::eColorAttachmentOptimal // Expected layout used by a subpass
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
		return device.createFramebuffer(vk::FramebufferCreateInfo{
			{}
			, imguiRenderPass
			, imageView // Attachments
			, surfaceExtent.width
			, surfaceExtent.height
			, 1 // Single layer, not doing stereo-rendering
		});
	};
	imguiFramebuffers = imguiSwapchainImageViews | std::views::transform(toFramebuffer) | std::ranges::to<std::vector>();
}
void VulkanApplication::initImGuiCommandPool()
{
	imguiCommandPool = device.createCommandPool(vk::CommandPoolCreateInfo{
		vk::CommandPoolCreateFlagBits::eResetCommandBuffer // reset and rerecord command buffer
		, getSuitableQueueFamilies(physicalDevice, surface).front().first
	});
}
void VulkanApplication::initImGuiCommandBuffer()
{
	imguiCommandBuffers = device.allocateCommandBuffers(vk::CommandBufferAllocateInfo{
		imguiCommandPool
		, vk::CommandBufferLevel::ePrimary // TODO: Secondary for UI? so we don't have to sync between 2 primary? or do we still need to sync between a primary and a secondary?
		, 1
	});
}
void VulkanApplication::cleanupImGui()
{
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
	ImGui_ImplVulkan_Shutdown();
}


// Must have an event to prevent data race between the imgui command buffer and the application rendr ocmmand buufer if they read/write to a descriptor
// TODO: Rebuild the swapchain if resizing
// TODO: Inflight frames
// Imgui is likely to share resource with the applicationss' commandbuffers commands, sync this imguiCommandbuffer and the application render commandbuffer
// Render the application first then pass the image rendered via color attachment to the top of the imgui renderpass to render the imgui ui ontop of the image
// Get the user inputs from imgui first then we will construct a frame with them
//void VulkanApplication::renderLoop(const CallbackRenderFunction& recordRenderingCommands, const ApplicationInfo& applicationInfo, const CallbackImguiFunction& imguiCommands, std::string_view windowName = "MyWindow")

// Always render in the order from the back to the front with respect to the viewport depth
void VulkanApplication::renderLoop(const RunInfo& runInfo, const ApplicationInfo& applicationInfo)
{
	const auto syncApplicationAndImguiEvent = device.createEvent(vk::EventCreateInfo{vk::EventCreateFlagBits::eDeviceOnly});

	while (!glfwWindowShouldClose(window))
	{
		// Pre submission
		glfwPollEvents();
		const auto preFrameRender = glfwGetTime();

		// Waiting for the previous submitted command buffers to finish and get new frame index
		const auto isFirstFrame = device.getFenceStatus(isCommandBufferExecutedFence) == vk::Result::eSuccess;
		std::ignore = device.waitForFences(isCommandBufferExecutedFence, VK_TRUE, std::numeric_limits<uint64_t>::max()); // Avoid modifying the command buffer when it's in used by the device
		device.resetFences(isCommandBufferExecutedFence);
		const auto resultValue = device.acquireNextImageKHR(swapchain, std::numeric_limits<uint64_t>::max(), isAcquiredImageReadSemaphore); // Semaphore will be raised when the acquired image is finished reading by the engine
		if (resultValue.result != vk::Result::eSuccess) throw std::runtime_error{"Failed to acquire the next image index."};
		const auto imageIndex = resultValue.value;

		// Application rendering commands
		const auto commandBuffer = commandBuffers.front();
		device.resetCommandPool(commandPool);
		commandBuffer.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
		runInfo.renderCommands(applicationInfo, imageIndex, isFirstFrame); // renderFrame = a function with command to be executed only?
		commandBuffer.setEvent(syncApplicationAndImguiEvent, vk::PipelineStageFlagBits::eBottomOfPipe);
		commandBuffer.end();

		// ImGui rendering commands
		ImGui_ImplVulkan_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();
		if (runInfo.imguiCommands.has_value()) runInfo.imguiCommands.value()();
		ImGui::Render();
		device.resetCommandPool(imguiCommandPool); // Reset the command buffers within this command pool, needed for the OneTimeSubmit flag
		const auto imguiCommandBuffer = imguiCommandBuffers.front();
		imguiCommandBuffer.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
		imguiCommandBuffer.waitEvents(syncApplicationAndImguiEvent, vk::PipelineStageFlagBits::eBottomOfPipe, vk::PipelineStageFlagBits::eTopOfPipe, {}, {}, {});
		const auto clearValues = static_cast<vk::ClearValue>(color::black);
		imguiCommandBuffer.beginRenderPass(vk::RenderPassBeginInfo{
			imguiRenderPass
			, imguiFramebuffers[imageIndex]
			, vk::Rect2D{vk::Offset2D{0, 0}, vk::Extent2D{surfaceExtent.width, surfaceExtent.height}}
			, clearValues // Doesn't matter because the imguiRenderPass load rather than clear the attachment. This is specifies in the attachmentDescription
		}, vk::SubpassContents::eInline);
		ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), imguiCommandBuffer); // Record imgui primitives into command buffer via imgui pipeline
		imguiCommandBuffer.endRenderPass();
		imguiCommandBuffer.end();

		// Submit
		const auto waitStages = std::vector<vk::PipelineStageFlags>{
			 vk::PipelineStageFlagBits::eColorAttachmentOutput // Wait for the imgui commands from the previous submission to finish
		};
		const auto submittedCommandBuffers = {commandBuffer, imguiCommandBuffer};
		queue.submit(vk::SubmitInfo{
			isAcquiredImageReadSemaphore // Wait for the image to be finished reading, then we will modify it via the commands in the commandBuffers
			, waitStages 
			, submittedCommandBuffers
			, isImageRenderedSemaphore // Raise when finished executing the commands (Sync with present queue)
		}, isCommandBufferExecutedFence); // Raise when finished executing the commands (Sync with host)

		// Present
		const auto presentResult = queue.presentKHR(vk::PresentInfoKHR{
			isImageRenderedSemaphore
			, swapchain
			, imageIndex
		});
		if (presentResult != vk::Result::eSuccess) throw std::runtime_error{"Failed to present image."};

		// Post submission
		const auto postFrameRender = glfwGetTime();
		const auto frameRenderTime = postFrameRender - preFrameRender;
		const auto framesPerSecond = static_cast<int>(std::round(1 / frameRenderTime));
		glfwSetWindowTitle(window, (std::string{runInfo.windowName} + " - FPS: " + std::to_string(framesPerSecond)).data());
	}

	device.waitIdle(); // Wait for all the fences to be unsignaled before clean up
	device.destroyEvent(syncApplicationAndImguiEvent);
}

void VulkanApplication::cleanUp() // Destroy the objects in reverse order of their creation order
{
	cleanupImGui();
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


