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
#include <tuple>
#include <cstdint> // Needed for uint32_t
#include <limits> // Needed for std::numeric_limits
#include <algorithm> // Needed for std::clamp
#include "VulkanApplication.h"
#include "Geometry.h"
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace
{
	// Enums
	enum class ShaderStage
	{
		vert // Vertex shader
		, tesc // Tessellation control shader
		, tese // Tessellation evaluation shader
		, geom // Geometry shadr
		, frag // Fragment shader
		, comp // Compute shader
	};
	using CurrentLastWriteTime = std::filesystem::file_time_type;
	using ShaderSourceProperties = std::tuple<std::filesystem::path, ShaderStage, CurrentLastWriteTime>;
	using QueueFamilyIndex = uint32_t;
	using QueuesPriorities = std::vector<float>;
	using QueueFamily = std::pair<QueueFamilyIndex, QueuesPriorities>;

	// Classes/Structs
	struct SurfaceAttributes
	{
		vk::SurfaceCapabilitiesKHR capabilities;
		std::vector<vk::SurfaceFormatKHR> formats;
		std::vector<vk::PresentModeKHR> presentModes;
	};

	// Helper functions
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
	[[nodiscard]] auto getShaderBinary(const std::string& path)
	{
		const auto filePath = std::filesystem::path{path};
		auto file = std::ifstream{filePath , std::ios::binary};
		if (!file.is_open()) throw std::runtime_error(std::string{"Can't open file at "} + filePath.string());
		auto shaderBinary = std::vector<char>(std::filesystem::file_size(filePath));
		file.read(shaderBinary.data(), shaderBinary.size());
		return shaderBinary;
	}
	[[nodiscard]] inline auto getEnumeration(const auto& container)
	{
		return std::views::zip(std::views::iota(0U, container.size()), container);
	}
	[[nodiscard]] inline auto unquotePathString(const std::filesystem::path& path) // Remove quotes returned by path.string()
	{
		auto stringStreamPath = std::stringstream{path.string()};
		auto unquotedPathString = std::string{};
		stringStreamPath >> std::quoted(unquotedPathString);
		return unquotedPathString;
	}
	[[nodiscard]] inline auto toString(const std::filesystem::file_time_type& writeTime)
	{
		auto sstream = std::stringstream{};
		auto writeTimeString = std::string{};
		sstream << writeTime;
		std::getline(sstream, writeTimeString);
		return writeTimeString;
	}
	[[nodiscard]] auto getShaderCompileCommand(const ShaderSourceProperties& shaderSourceProperties)
	{
		const auto& [shaderPathRelative, shaderStage, _] = shaderSourceProperties;
		const auto shaderPath = std::filesystem::absolute(shaderPathRelative);
		const auto compilerPath = std::filesystem::absolute(std::filesystem::path("Dependencies/glslc.exe"));
		const auto binaryPath = std::filesystem::absolute(std::filesystem::path("Binaries/Shader/")/(shaderPath.stem().string() + std::string{".spv"}));
		auto shaderStageName = std::string{};
		switch (shaderStage)
		{
			case ShaderStage::vert: shaderStageName = "vert"; break;
			case ShaderStage::tese: shaderStageName = "tese"; break;
			case ShaderStage::tesc: shaderStageName = "tesc"; break;
			case ShaderStage::geom: shaderStageName = "geom"; break;
			case ShaderStage::frag: shaderStageName = "frag"; break;
			case ShaderStage::comp: shaderStageName = "comp"; break;
		};
		auto command = std::stringstream{};
		const auto unquotedShaderPathString = unquotePathString(shaderPath);
		const auto unquotedCompilerPathString = unquotePathString(compilerPath);
		const auto unquotedBinaryPathString = unquotePathString(binaryPath);
		command << unquotedCompilerPathString << " " << "-fshader-stage=" << shaderStageName << " " << unquotedShaderPathString << " -o " << unquotedBinaryPathString;
		return command.str();
	}
	void validateShaders()
	{
		const auto shaderSourcesProperties = std::vector<ShaderSourceProperties>{
			{"Shader/GeneralVertex.glsl", ShaderStage::vert, std::filesystem::last_write_time("Shader/GeneralVertex.glsl")}
			, {"Shader/GeneralFragment.glsl", ShaderStage::frag, std::filesystem::last_write_time("Shader/GeneralFragment.glsl")}
		};
		const auto lastWriteTimesFilePath = std::filesystem::path{"Shader/LastWriteTimes.txt"};
		auto lastWriteTimesFile = std::fstream{};
		if (!std::filesystem::exists(lastWriteTimesFilePath)) lastWriteTimesFile = std::fstream{lastWriteTimesFilePath, std::ios::out}; // Create a file to write the write times into.
		lastWriteTimesFile = std::fstream{lastWriteTimesFilePath, std::ios::in | std::ios::out}; // Reopen in read and write modes. Using open() doesn't work as expected sometimes, assigning new fstreams instead
		const auto compileAndWriteToFile = [&](const ShaderSourceProperties& shaderSourceProperties)
		{
			const auto& [shaderPath, _, currentLastWriteTime] = shaderSourceProperties;
			const auto compileCommand = getShaderCompileCommand(shaderSourceProperties);
			std::cout << tag::warning << "Compiling shader at " << shaderPath << '\n';
			if (std::system(compileCommand.data()) == 0)
			{
				std::cout << tag::warning << "Finished\n";
				lastWriteTimesFile << currentLastWriteTime << std::endl;
			}
			else lastWriteTimesFile << "Failed to compile shader" << std::endl;
		};
		if (std::filesystem::is_empty(lastWriteTimesFilePath))
		{
			std::ranges::for_each(shaderSourcesProperties, compileAndWriteToFile);
			return;
		}
		auto writeTime = std::string{};
		auto lastWriteTimes = std::vector<std::string>{};
		while (std::getline(lastWriteTimesFile, writeTime)) lastWriteTimes.push_back(writeTime);
		lastWriteTimesFile = std::fstream{lastWriteTimesFilePath, std::ios::out | std::ios::trunc}; // Truncate mode will clear the file first for writting.
		if (lastWriteTimes.size() < shaderSourcesProperties.size()) // New shaders need to be compiled whose last write time hasn't been captured
		{
			const auto different = shaderSourcesProperties.size() - lastWriteTimes.size();
			for (size_t i = 0; i < different; i++) lastWriteTimes.push_back("New shader");
		}
		for (const auto& [lastWriteTime, shaderSourceProperties] : std::views::zip(lastWriteTimes, shaderSourcesProperties))
		{
			const auto& [shaderPath, _, currentLastWriteTime] = shaderSourceProperties;
			if (lastWriteTime != toString(currentLastWriteTime)) compileAndWriteToFile(shaderSourceProperties);
			else
			{
				std::cout << tag::log << "Shader at " << shaderPath << " has no modifications\n";
				lastWriteTimesFile << currentLastWriteTime << std::endl;
			}
		}
	}
}

VulkanApplication::VulkanApplication()
	: window{nullptr}
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
	, vertexBuffer{}
	, pipelineLayout{}
	, graphicPipeline{}
	, framebuffers{}
	, commandPool{}
	, commandBuffers{}
	, isPresentationEngineReadFinished{}
	, isImageRendered{}
	, isPreviousImagePresented{}
{
}
VulkanApplication::~VulkanApplication()
{
	// Destroy the objects in reverse order of their creation order
	device.destroy(isPresentationEngineReadFinished);
	device.destroy(isImageRendered);
	device.destroy(isPreviousImagePresented);
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

void VulkanApplication::run() noexcept
{
	try
	{
		validateShadersWorker = std::thread{validateShaders};
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
	const auto vertexShaderBinary = getShaderBinary("Binaries/Shader/GeneralVertex.spv");
	const auto vertexShaderModuleCreateInfo = vk::ShaderModuleCreateInfo{{}, vertexShaderBinary.size(), (uint32_t*)(vertexShaderBinary.data())};
	const auto vertexShaderModule = device.createShaderModule(vertexShaderModuleCreateInfo);
	const auto vertexShaderStageCreateInfo = vk::PipelineShaderStageCreateInfo{
		{}
		, vk::ShaderStageFlagBits::eVertex
		, vertexShaderModule
		, "main" // entry point
	};
	// Fragment shader stage
	const auto fragmentShaderBinary = getShaderBinary("Binaries/Shader/GeneralFragment.spv");
	const auto fragmentCreateInfo = vk::ShaderModuleCreateInfo{{}, fragmentShaderBinary.size(), (uint32_t*)(fragmentShaderBinary.data())};
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
	// NDC space
	const auto bindingNumber = 0U;
	const auto triangle = std::vector<Vertex>{
		{{0.5f, 0.5f, 0.0f}, {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}}
		, {{-0.5f, 0.5f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}}
		, {{0.0f, -0.5f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}}
	};
	//const auto totalBytesSize = triangle.size() * sizeof(Vertex);
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
	// Logical buffer
	const auto queueFamilies = getSuitableQueueFamilies(physicalDevice, surface);
	const auto toQueueFamilyIndex = [](const QueueFamily& queueFamily){return queueFamily.first;};
	const auto queueFamiliesIndex = queueFamilies | std::views::transform(toQueueFamilyIndex) | std::ranges::to<std::vector>();
	const auto bufferCreateInfo = vk::BufferCreateInfo{
		{}
		, triangle.size() * sizeof(Vertex)
		, vk::BufferUsageFlagBits::eVertexBuffer
		, vk::SharingMode::eExclusive
		, queueFamiliesIndex
	};
	vertexBuffer = device.createBuffer(bufferCreateInfo);
	// Physical memory allocation for the logical buffer
	const auto memoryRequirements = device.getBufferMemoryRequirements(vertexBuffer);
	const auto memoryProperties = physicalDevice.getMemoryProperties();
	const auto indexedMemoryTypes = getEnumeration(memoryProperties.memoryTypes);
	const auto isSuitable = [&](const auto& indexedMemoryType)
	{
		const auto& [memoryIndex, memoryType] = indexedMemoryType;
		const auto memoryTypeBits = (1 << memoryIndex); // The type represented as bits, each type is counted as a power of 2 from 0
		const auto hasRequiredMemoryType = memoryRequirements.memoryTypeBits & memoryTypeBits;
		const auto hasRequiredMemoryProperty = memoryType.propertyFlags & (vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
		return hasRequiredMemoryType && hasRequiredMemoryProperty;
	};
	const auto iterIndexedMemoryTypes = std::ranges::find_if(indexedMemoryTypes, isSuitable);
	if (iterIndexedMemoryTypes == indexedMemoryTypes.end()) throw std::runtime_error{"Failed to find suitable memory type"};
	const auto memoryAllocateInfo = vk::MemoryAllocateInfo{
		memoryRequirements.size
		, std::get<0>(*iterIndexedMemoryTypes)
	};
	vertexBufferMemory = device.allocateMemory(memoryAllocateInfo);
	// Bind the logical buffer to the allocated memory
	device.bindBufferMemory(vertexBuffer, vertexBufferMemory, 0);
	// Fill-in the allocated memory with buffer's data
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
void VulkanApplication::initSyncObjects()
{
	const auto semCreateInfo = vk::SemaphoreCreateInfo{};
	const auto fenceCreateInfo = vk::FenceCreateInfo{vk::FenceCreateFlagBits::eSignaled}; // First frame doesn't have to wait for the unexisted previous image
	isPresentationEngineReadFinished = device.createSemaphore(semCreateInfo);
	isImageRendered = device.createSemaphore(semCreateInfo);
	isPreviousImagePresented = device.createFence(fenceCreateInfo);
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

void VulkanApplication::drawFrame()
{
	// Get framebufer
	std::ignore = device.waitForFences(isPreviousImagePresented, VK_TRUE, UINT64_MAX); // Block until fences is signaled
	device.resetFences(isPreviousImagePresented);
	const auto resultValue = device.acquireNextImageKHR(swapchain, UINT64_MAX, isPresentationEngineReadFinished, VK_NULL_HANDLE);

	// Record and submit commandbuffer for that image
	const auto imageIndex = resultValue.value;
	auto& commandBuffer = commandBuffers.front();
	recordCommandBuffer(commandBuffer, imageIndex);
	const auto stages = {vk::PipelineStageFlagBits::eColorAttachmentOutput};
	const auto waitStages = stages | std::ranges::to<std::vector<vk::PipelineStageFlags>>(); 
	auto submitInfo = vk::SubmitInfo{isPresentationEngineReadFinished, waitStages, commandBuffer, isImageRendered};
	queue.submit(submitInfo, isPreviousImagePresented);

	// Present
	const auto presentInfo = vk::PresentInfoKHR{isImageRendered, swapchain, imageIndex};
	std::ignore = queue.presentKHR(presentInfo);
}


