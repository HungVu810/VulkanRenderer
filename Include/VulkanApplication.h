#pragma once
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>
#include <iostream>
#include <vector>
#include <thread>
#include <unordered_map>
#include "Prompt.h"
#include "Shader.h"

#ifdef NDEBUG
	const bool bEnableValidationLayers = false;
#else
	const bool isValidationLayersEnabled = true;
#endif

constexpr auto width = uint32_t{800};
constexpr auto height = uint32_t{600};

class VulkanApplication
{
public:
	VulkanApplication();

	~VulkanApplication();

	// All exceptions are handled in this function so we can clean up the
	// resources thereafter.
	void run() noexcept;

private:
	void initWindow();

	void initDispatcher();
	void initInstance();
	void initDebugMessenger();
	void initSurface();
	void initPhysicalDevice();
	void initDevice();
	void initQueue();
	void initSwapChain();
	void initImageViews();
	void initRenderPass();
	void initGraphicPipeline();
	void initVulkan();
	void initFrameBuffer();
	void initCommandPool();
	void initCommandBuffer();
	void initSyncObjects();
	void recordCommandBuffer(vk::CommandBuffer& commandBuffer, uint32_t imageIndex);

	void mainLoop();

	void drawFrame();

	void cleanUp();

	static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
		VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
		VkDebugUtilsMessageTypeFlagsEXT messageType,
		const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
		void* pUserData)
	{
		const auto message = std::string{ pCallbackData->pMessage };
		const auto posError = message.find("Error");
		assertm(posError == std::string::npos, message); // This is likely a programming error
		const auto posWarning = message.find("Warning");
		const auto tagToUse = posWarning == std::string::npos ? tag::validation : tag::warning;
		std::cerr << tagToUse << message << '\n';
		return VK_FALSE;
	}

private:
	GLFWwindow* window;
	std::thread validateShadersWorker;
	vk::Instance instance;
	vk::DebugUtilsMessengerCreateInfoEXT debugMessengerCreateInfo; // needed ?
	vk::DebugUtilsMessengerEXT debugMessenger;
	vk::SurfaceKHR surface;
	vk::PhysicalDevice physicalDevice;
	vk::Device device;
	vk::Queue queue;
	vk::SurfaceFormatKHR surfaceFormat;
	vk::Extent2D surfaceExtent;
	vk::SwapchainKHR swapchain;
	std::vector<vk::ImageView> swapchainImageViews;
	vk::RenderPass renderPass;
	std::unordered_map<std::string, Shader> shaderMap;
	vk::Buffer vertexBuffer;
	vk::DeviceMemory vertexBufferMemory;
	vk::PipelineLayout pipelineLayout;
	vk::Pipeline graphicPipeline;
	std::vector<vk::Framebuffer> framebuffers;
	vk::CommandPool commandPool;
	std::vector<vk::CommandBuffer> commandBuffers;
	vk::Semaphore isPresentationEngineReadFinished;
	vk::Semaphore isImageRendered;
	vk::Fence isPreviousImagePresented;
};


