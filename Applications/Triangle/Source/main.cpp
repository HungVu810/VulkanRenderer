#include "Geometry.h"
#include "Shader.h"
#include "Allocation.h"

#include <VulkanApplication.h>

#include <cstdint> // Needed for uint32_t
#include <thread> // In case of wanting more workers
#include <limits> // Needed for std::numeric_limits



// TODO: Create a seperate trianngle applicatoin.h, only puut some common headers in the CMAKELIST

	, swapchainImageViews{}
	, renderPass{}
	, shaderMap{}
	, vertexBuffer{}
	, pipelineLayout{}
	, graphicPipeline{}
	, framebuffers{}


// triangle, pre render loop
	validateShaders(shaderMap); // Only validate shader passed through run()
void initImageViews();
void initRenderPass();
void initGraphicPipeline();
void initFrameBuffer();
// triangle
// ========== useGraphicPipeline ==========
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
		, vk::ImageLayout::eUndefined // Expected layout at the beginning of the renderpass, undefined is the default
		, vk::ImageLayout::ePresentSrcKHR // Expected layout at the end of the renderpass, final layout for presentationo
	};
	const auto attachmentIndex = 0U; // TODO: Make a type that associated the index and the created attachmentDescription, ie vector<attachmentDescription>

	// Subpass description
	const auto attachmentReference = vk::AttachmentReference{
		attachmentIndex
		, vk::ImageLayout::eColorAttachmentOptimal // Expected layout at the end of this subpass
	};
	const auto subpassDescription = vk::SubpassDescription{
		{}
		, vk::PipelineBindPoint::eGraphics
		, {}
		, attachmentReference
	};

	const auto subpassDependency = vk::SubpassDependency{
		VK_SUBPASS_EXTERNAL // srcSubpass, the one beforer
		, 0U // dstSubpass, the current one
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

// --------- DYNAMIC-STATE
	// Modify a subset of options of the fixed states without recreating the pipeline
	const auto dynamicStateCreateInfo = vk::PipelineDynamicStateCreateInfo{};
// --------- DYNAMIC-STATE

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


 // This is not the default so make it into a seperate application as a triangle demo
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
	std::ignore = device.waitForFences(isCommandBufferExecutedFence, VK_TRUE, std::numeric_limits<uint64_t>::max());
	device.resetFences(isCommandBufferExecutedFence);
	const auto resultValue = device.acquireNextImageKHR(swapchain, std::numeric_limits<uint64_t>::max(), isAcquiredImageReadSemaphore, VK_NULL_HANDLE);

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

void VulkanApplication::renderLoop()
{
	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();
		drawFrame();
	}

	device.waitIdle(); // wait for the queue(s) to become idle, finished executing the cmd?
}


// ========== useGraphicPipeline ==========

	//if (APPLICATION_USES_GRAPHIC_PIPELINE)
	//{
	//	for (const vk::Framebuffer& framebuffer : framebuffers) device.destroyFramebuffer(framebuffer);
	//	device.destroyPipeline(graphicPipeline);
	//	device.destroyPipelineLayout(pipelineLayout);
	//	device.freeMemory(vertexBufferMemory);
	//	device.destroyBuffer(vertexBuffer);
	//	device.destroyRenderPass(renderPass);
	//	for (const vk::ImageView& imageView : swapchainImageViews) device.destroyImageView(imageView);
	//}
	//else
	//{
	//}

// render loop
void mainLoop();
void recordCommandBuffer(vk::CommandBuffer& commandBuffer, uint32_t imageIndex);
void drawFrame();


	std::vector<vk::ImageView> swapchainImageViews;
	vk::RenderPass renderPass;
	vk::Buffer vertexBuffer;
	vk::DeviceMemory vertexBufferMemory;
	vk::PipelineLayout pipelineLayout;
	vk::Pipeline graphicPipeline;
	std::vector<vk::Framebuffer> framebuffers;

int main()
{
	VulkanApplication app;
	app.run();
	return EXIT_SUCCESS;
}


