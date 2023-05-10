#include <thread> // In case of wanting more workers
#include <cstdint> // Needed for uint32_t
#include <limits> // Needed for std::numeric_limits
#include <fstream>
#include <filesystem>
#include <vector>
#include <cassert>
#include "VulkanApplication.h"
#include "Shader.h"
#include "Allocation.h"
#include "vulkan/vulkan.hpp" // Do not place this header above the VulkanApplication.h

#define APPLICATION_INFO_BINDINGS const auto& [surface, physicalDevice, device, queue, queueFamilies, swapchain, surfaceFormat, surfaceExtent, renderCommandBuffers, isAcquiredImageReadSemaphore, isImageRenderedSemaphore, isRenderCommandBufferExecutedFence] = applicationInfo;

namespace 
{
	using Intensity = float; // Can't do short because sampler3D will always return a vec of floats
	constexpr auto NUM_SLIDES = 113;
	constexpr auto SLIDE_HEIGHT = 256;
	constexpr auto SLIDE_WIDTH = 256;
	constexpr auto NUM_INTENSITIES = NUM_SLIDES * SLIDE_HEIGHT * SLIDE_WIDTH;
	constexpr auto TOTAL_SCAN_BYTES = NUM_INTENSITIES * sizeof(Intensity);

	// z-y-x order, contains all intensity values of each slide images. Tightly packed
	// vector instead of array bececause NUM_INTENSITIES is large, occupy the heap instead
	auto intensities = Resource{std::vector<Intensity>(NUM_INTENSITIES), toVulkanFormat<Intensity>()}; // Check 
	auto shaderMap = std::unordered_map<std::string, Shader>{};
	vk::Image raycastedImage; vk::DeviceMemory raycastedImageMemory;
	vk::Image volumeImage; vk::DeviceMemory volumeImageMemory;
	vk::Buffer stagingBuffer; vk::DeviceMemory stagingBufferMemory;
	vk::ImageView volumeImageView; vk::ImageView raycastedImageView;
	vk::Sampler volumeImageSampler;
	vk::DescriptorSetLayout descriptorSetLayout;
	vk::DescriptorPool descriptorPool;
	std::vector<vk::DescriptorSet> descriptorSets;
	vk::ShaderModule volumeShaderModule;
	vk::PipelineLayout computePipelineLayout;
	vk::Pipeline computePipeline;

	// Private helperrs
	void loadVolumeData()
	{
		auto dataIndex = 0;
		for (auto slideIndex = 1; slideIndex <= NUM_SLIDES; slideIndex++)
		{
			const auto ctPath = std::filesystem::path{VOLUME_DATA"/CThead." + std::to_string(slideIndex)};
			auto ctFile = std::ifstream{ctPath, std::ios_base::binary};
			if (!ctFile) throw std::runtime_error{"Can't open file at " + ctPath.string()};

			auto intensity = uint16_t{0}; // Data is type short
			for (; ctFile.read(reinterpret_cast<char*>(&intensity), sizeof(intensity)); dataIndex++)
			{
				// Swap byte order if running on little-endian system
				#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
					data = (data >> 8) | (data << 8);
				#endif
				intensities.data[dataIndex] = intensity;
			}
		}
		// Normalize the data
		const auto iterMax = std::ranges::max_element(intensities.data);
		const auto maxIntensity = *iterMax;
		const auto normalize = [&](Intensity& intensity){ return intensity /= maxIntensity; };
		std::ranges::for_each(intensities.data, normalize);
		const auto iterMaxlc = std::ranges::max_element(intensities.data);
	}
	void initComputePipeline(const ApplicationInfo& applicationInfo)
	{
		APPLICATION_INFO_BINDINGS

		// Format check
		checkFormatFeatures(physicalDevice, surfaceFormat, vk::FormatFeatureFlagBits::eStorageImage);
		checkFormatFeatures(physicalDevice, surfaceFormat, vk::FormatFeatureFlagBits::eTransferDst);
		checkFormatFeatures(physicalDevice, surfaceFormat, vk::FormatFeatureFlagBits::eTransferSrc);

		checkFormatFeatures(physicalDevice, intensities.format, vk::FormatFeatureFlagBits::eSampledImage);
		checkFormatFeatures(physicalDevice, intensities.format, vk::FormatFeatureFlagBits::eTransferDst);

		const auto queueFamiliesIndex = getQueueFamilyIndices(queueFamilies);

		// Reserve device memory for the raycasted image
		raycastedImage = device.createImage(vk::ImageCreateInfo{
			{}
			, vk::ImageType::e2D
			, surfaceFormat
			, vk::Extent3D{surfaceExtent, 1}
			, 1 // The only mip level for this image
			, 1 // Single layer, no stereo-rendering
			, vk::SampleCountFlagBits::e1 // One sample, no multisampling
			, vk::ImageTiling::eOptimal
			, vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc // Transfer command from the staging device memory to the image memory via cmdCopy
			, vk::SharingMode::eExclusive
			, queueFamiliesIndex
		});
		raycastedImageMemory = allocateMemory(device, physicalDevice, device.getImageMemoryRequirements(raycastedImage), false); // TODO: WHY DOESN'T THIS ALLOW HOST-VISIBLE MEMORY?
		device.bindImageMemory(raycastedImage, raycastedImageMemory, 0);

		// Reserve device memory for volume image
		const auto volumeImageCreateInfo =
		volumeImage = device.createImage(vk::ImageCreateInfo{
			{}
			, vk::ImageType::e3D
			, intensities.format
			, {SLIDE_WIDTH, SLIDE_HEIGHT, NUM_SLIDES}
			, 1 // The only mip level for this image
			, 1 // Single layer, no stereo-rendering
			, vk::SampleCountFlagBits::e1 // One sample, no multisampling
			, vk::ImageTiling::eOptimal
			, vk::ImageUsageFlagBits::eSampled // Sampled by the compute shader
				| vk::ImageUsageFlagBits::eTransferDst // Volumn data is transfered from a staging buffer transferSrc
			, vk::SharingMode::eExclusive
			, queueFamiliesIndex
		});
		volumeImageMemory = allocateMemory(device, physicalDevice, device.getImageMemoryRequirements(volumeImage), false); // TODO: WHY DOESN'T THIS ALLOW HOST-VISIBLE MEMORY?
		device.bindImageMemory(volumeImage, volumeImageMemory, 0);

		// Reserve device memory for staging buffer
		stagingBuffer = device.createBuffer(vk::BufferCreateInfo{
			{}
			, TOTAL_SCAN_BYTES
			, vk::BufferUsageFlagBits::eTransferSrc // Store and transfer volume data to volume image stored on the device memory via a copy command
			, vk::SharingMode::eExclusive
			, queueFamiliesIndex
		});
		const auto stagingBufferMemoryRequirement = device.getBufferMemoryRequirements(stagingBuffer);
		stagingBufferMemory = allocateMemory(device, physicalDevice, stagingBufferMemoryRequirement, true); // TODO: WHY THIS ALLOW HOST-VISIBLE MEMORY?
		device.bindBufferMemory(stagingBuffer, stagingBufferMemory, 0);

		// Upload volume data to the staging buffer, we will transfer this staging buffer data over to the volume image later
		void* memory = device.mapMemory(stagingBufferMemory, 0, stagingBufferMemoryRequirement.size);
		std::memcpy(memory, intensities.data.data(), TOTAL_SCAN_BYTES);
		device.unmapMemory(stagingBufferMemory);

		// Create a descriptor for the volumn image and the raycasted image
		const auto maxDescriptorSets = 1;	
		const auto descriptorPoolSizes = std::vector<vk::DescriptorPoolSize>{
			{vk::DescriptorType::eCombinedImageSampler, 1} // Volume image
			, {vk::DescriptorType::eStorageImage, 1} // Raycasted image
		};
		const auto descriptorPoolCreateInfo = vk::DescriptorPoolCreateInfo{
			{}
			, maxDescriptorSets
			, descriptorPoolSizes
		};
		descriptorPool = device.createDescriptorPool(descriptorPoolCreateInfo);
		volumeImageSampler = device.createSampler(vk::SamplerCreateInfo{
			{}
			, vk::Filter::eNearest // TODO: try linear
			, vk::Filter::eNearest // TODO: try linear
			, vk::SamplerMipmapMode::eNearest
			, vk::SamplerAddressMode::eClampToBorder // U
			, vk::SamplerAddressMode::eClampToBorder // V
			, vk::SamplerAddressMode::eClampToBorder // W
			, 0.0f
			, VK_FALSE
			, 0.0f
			, VK_FALSE
			, vk::CompareOp::eNever
			, 0.0f
			, 0.0f
			, vk::BorderColor::eIntOpaqueBlack // Border Color
			, VK_FALSE // Always normalize coordinate

		});
		const auto volumnImageBinding = vk::DescriptorSetLayoutBinding{
			0
			, vk::DescriptorType::eCombinedImageSampler
			, vk::ShaderStageFlagBits::eCompute
			, volumeImageSampler
		};
		const auto raycastedImageBinding = vk::DescriptorSetLayoutBinding{
			1
			, vk::DescriptorType::eStorageImage
			, vk::ShaderStageFlagBits::eCompute
			, volumeImageSampler
		};
		const auto layoutBindings = {volumnImageBinding, raycastedImageBinding};
		descriptorSetLayout = device.createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo{{}, layoutBindings});
		descriptorSets = device.allocateDescriptorSets(vk::DescriptorSetAllocateInfo{descriptorPool, descriptorSetLayout});

		// Bind the volume image view to the descriptor
		volumeImageView = device.createImageView(vk::ImageViewCreateInfo{
			{}
			, volumeImage
			, vk::ImageViewType::e3D
			, intensities.format
			, {vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity}
			, vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
		});
		auto descriptorImageInfo = vk::DescriptorImageInfo{
			volumeImageSampler
			, volumeImageView
			, vk::ImageLayout::eShaderReadOnlyOptimal // Expected layout of the descriptor
		};
		device.updateDescriptorSets(vk::WriteDescriptorSet{
			descriptorSets.front()
			, 0
			, 0
			, vk::DescriptorType::eCombinedImageSampler
			, descriptorImageInfo
		}, {});

		// Bind the raycasted image view to the descriptor
		raycastedImageView = device.createImageView(vk::ImageViewCreateInfo{
			{}
			, raycastedImage
			, vk::ImageViewType::e2D
			, surfaceFormat
			, {vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity}
			, vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
		});
		descriptorImageInfo = vk::DescriptorImageInfo{
			volumeImageSampler
			, raycastedImageView
			, vk::ImageLayout::eGeneral // Expected layout of the storage descriptor
		};
		device.updateDescriptorSets(vk::WriteDescriptorSet{
			descriptorSets.front()
			, 1
			, 0
			, vk::DescriptorType::eStorageImage
			, descriptorImageInfo
		}, {});

		// Compute pipeline
		const auto volumeShaderBinaryData = getShaderBinaryData(shaderMap, "VolumeRendering.comp");
		volumeShaderModule = device.createShaderModule(vk::ShaderModuleCreateInfo{{}, volumeShaderBinaryData});
		const auto computeShaderStageCreateInfo = vk::PipelineShaderStageCreateInfo{
			{}
			, vk::ShaderStageFlagBits::eCompute
			, volumeShaderModule
			, "main"
		};
		computePipelineLayout = device.createPipelineLayout(vk::PipelineLayoutCreateInfo{{}, descriptorSetLayout});
		const auto resultValue = device.createComputePipeline({}, vk::ComputePipelineCreateInfo{
			{}
			, computeShaderStageCreateInfo
			, computePipelineLayout
		});
		if (resultValue.result != vk::Result::eSuccess) throw std::runtime_error{"Unable to create a compute pipeline."};
		computePipeline = resultValue.value;
	}
	void submitTemporaryCommandBuffer(const ApplicationInfo& applicationInfo)
	{
		// Purpose of this temporary commandBuffer submission:
		// Layout transition of the swapchain images to presentSrcKHR (PRESENTABLE IMAGE) in order to acquire the image index during the rendering loop without causing error

		APPLICATION_INFO_BINDINGS

		const auto queueFamilyIndex = getQueueFamilyIndices(queueFamilies).front(); // TODO: Use the first queueFamilyIndex for now

		const auto commandPool = device.createCommandPool(vk::CommandPoolCreateInfo{
			vk::CommandPoolCreateFlagBits::eTransient
			, queueFamilyIndex
		});
		const auto commandBuffers = device.allocateCommandBuffers(vk::CommandBufferAllocateInfo{
			commandPool
			, vk::CommandBufferLevel::ePrimary
			, 1
		});
		const auto commandBuffer = commandBuffers.front();

		// Recording
		commandBuffer.begin(vk::CommandBufferBeginInfo{
			vk::CommandBufferUsageFlagBits::eOneTimeSubmit
		});

		// Swapchain images layout: undefined -> presentSrcKHR
		const auto swapchainImages = device.getSwapchainImagesKHR(swapchain);
		for (const auto& image : swapchainImages)
		{
			commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eAllCommands, {}, {}, {}, vk::ImageMemoryBarrier{
				vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite
				, vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite
				, vk::ImageLayout::eUndefined
				, vk::ImageLayout::ePresentSrcKHR
				, VK_QUEUE_FAMILY_IGNORED // Same queue family, don't transfer the queue ownership
				, VK_QUEUE_FAMILY_IGNORED
				, image
				, vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
			});
		}

		// End recording
		commandBuffer.end();

		// Submit to transfer queue
		queue.submit({}, {});
		queue.waitIdle();

		device.freeCommandBuffers(commandPool, commandBuffer);
		device.destroyCommandPool(commandPool);
	}
	void recordRenderCommandBuffer(const ApplicationInfo& applicationInfo, uint32_t imageIndex, vk::Result fenceResult)
	{
		APPLICATION_INFO_BINDINGS

		const auto& renderCommandBuffer = renderCommandBuffers.front(); // TODO: Use the first command buffer for now

		renderCommandBuffer.begin(vk::CommandBufferBeginInfo{});

		if (fenceResult == vk::Result::eSuccess) // One time only
		{
			// Volume image layout: undefined -> transferDstOptimal, which is the expected layout when using the copy command
			renderCommandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, {vk::ImageMemoryBarrier{
				vk::AccessFlagBits::eNone
				, vk::AccessFlagBits::eTransferWrite
				, vk::ImageLayout::eUndefined
				, vk::ImageLayout::eTransferDstOptimal
				, VK_QUEUE_FAMILY_IGNORED // Same queue family, don't transfer the queue ownership
				, VK_QUEUE_FAMILY_IGNORED
				, volumeImage
				, vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
			}});

			renderCommandBuffer.copyBufferToImage(stagingBuffer, volumeImage, vk::ImageLayout::eTransferDstOptimal, vk::BufferImageCopy{
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
			});

			// Volume image layout: transferDstOptimal -> shaderReadOnlyOptimal
			renderCommandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eBottomOfPipe, {}, {}, {}, {vk::ImageMemoryBarrier{
				vk::AccessFlagBits::eTransferWrite
				, vk::AccessFlagBits::eNone
				, vk::ImageLayout::eTransferDstOptimal
				, vk::ImageLayout::eShaderReadOnlyOptimal
				, VK_QUEUE_FAMILY_IGNORED // Same queue family, don't transfer the queue ownership
				, VK_QUEUE_FAMILY_IGNORED
				, volumeImage
				, vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
			}});
		}

		// Transition layout of the image descriptors before compute pipeline
		// Raycasted image layout: undefined -> general, expected by the descriptor
		renderCommandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eComputeShader, {}, {}, {}, vk::ImageMemoryBarrier{
				vk::AccessFlagBits::eNone
				, vk::AccessFlagBits::eShaderWrite
				, vk::ImageLayout::eUndefined // Default & discard the previous contents of the raycastedImage
				, vk::ImageLayout::eGeneral
				, VK_QUEUE_FAMILY_IGNORED // Same queue family, don't transfer the queue ownership
				, VK_QUEUE_FAMILY_IGNORED
				, raycastedImage
				, vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
		});

		renderCommandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, computePipeline);
		renderCommandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, computePipelineLayout, 0, descriptorSets, {}); // 3D volume data
		const auto numInvocationPerX = 10; // Also put this number in the compute shader
		const auto numInvocationPerY = 10; // Also put this number in the compute shader
		assert((WIDTH % numInvocationPerX) == 0 && (HEIGHT % numInvocationPerY) == 0);
		renderCommandBuffer.dispatch(WIDTH / numInvocationPerX, HEIGHT / numInvocationPerY, 1); // *******NOTE: group size must be at least 1

		const auto swapchainImage = device.getSwapchainImagesKHR(swapchain)[imageIndex];

		// Barrier to sync writting to raycastedImage via compute shader and copy it to swapchain image
		// Raycasted image layout: general -> transferSrc, before copy commmand
		// Swapchain image layout: undefined -> transferDst, before copy command
		renderCommandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, {vk::ImageMemoryBarrier{
			vk::AccessFlagBits::eShaderWrite // Compute shader writes to raycasted image, vk::AccessFlagBits::eShaderWrite, only use this when the shader write to the memory
			, vk::AccessFlagBits::eTransferRead // Wait until raycasted image is finished written to then copy
			, vk::ImageLayout::eGeneral
			, vk::ImageLayout::eTransferSrcOptimal
			, VK_QUEUE_FAMILY_IGNORED // Same queue family, don't transfer the queue ownership
			, VK_QUEUE_FAMILY_IGNORED // Same queue family, don't transfer the queue ownership
			, raycastedImage
			, vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
		}, vk::ImageMemoryBarrier{
			vk::AccessFlagBits::eNone
			, vk::AccessFlagBits::eTransferWrite // Wait until raycasted image is finished written to then copy
			, vk::ImageLayout::eUndefined // Default & discard the previous contents of the swapchainImage
			, vk::ImageLayout::eTransferDstOptimal
			, VK_QUEUE_FAMILY_IGNORED // Same queue family, don't transfer the queue ownership
			, VK_QUEUE_FAMILY_IGNORED // Same queue family, don't transfer the queue ownership
			, swapchainImage
			, vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
		}});

		renderCommandBuffer.copyImage(raycastedImage, vk::ImageLayout::eTransferSrcOptimal, swapchainImage, vk::ImageLayout::eTransferDstOptimal, vk::ImageCopy{
				vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, 0, 0, 1}
				, {0, 0, 0}
				, vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, 0, 0, 1}
				, {0, 0, 0}
				, vk::Extent3D{surfaceExtent, 1}
		});

		// Transfer the swapchain image layout back to a presentable layout
		// Swapchain image layout: transferDst -> presentSrcKHR, before presented
		renderCommandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eBottomOfPipe, {}, {}, {}, vk::ImageMemoryBarrier{
				vk::AccessFlagBits::eTransferWrite // written during the copy command
				, vk::AccessFlagBits::eNone
				, vk::ImageLayout::eTransferDstOptimal
				, vk::ImageLayout::ePresentSrcKHR
				, VK_QUEUE_FAMILY_IGNORED
				, VK_QUEUE_FAMILY_IGNORED
				, swapchainImage
				, vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
		});

		renderCommandBuffer.end();
	}

	// Private render functions
	void preRenderLoop(const ApplicationInfo& applicationInfo)
	{
		loadVolumeData();
		validateShaders(shaderMap);
		initComputePipeline(applicationInfo);
		submitTemporaryCommandBuffer(applicationInfo);;
	}
	void renderFrame(const ApplicationInfo& applicationInfo)
	{
		APPLICATION_INFO_BINDINGS
	
		const auto fenceResult = device.getFenceStatus(isRenderCommandBufferExecutedFence); // Only signaled in the first loop, we will use this to make proper layout transition when record the render command buffer
	
		// Fence and submission
		std::ignore = device.waitForFences(isRenderCommandBufferExecutedFence, VK_TRUE, std::numeric_limits<uint64_t>::max()); // Avoid modifying the command buffer when it's in used by the device
		device.resetFences(isRenderCommandBufferExecutedFence);
	
		const auto resultValue = device.acquireNextImageKHR(swapchain, std::numeric_limits<uint64_t>::max(), isAcquiredImageReadSemaphore); // Semaphore will be raised when the acquired image is finished reading by the engine
		if (resultValue.result != vk::Result::eSuccess) throw std::runtime_error{"Failed to acquire the next image index."};
	
		const auto imageIndex = resultValue.value;
		const auto waitStages = std::vector<vk::PipelineStageFlags>{vk::PipelineStageFlagBits::eComputeShader}; // Wait the compute shader if we have inflight command buffers
		recordRenderCommandBuffer(applicationInfo, imageIndex, fenceResult);
		queue.submit(vk::SubmitInfo{
			isAcquiredImageReadSemaphore // Wait for the image to be finished reading, then we will modify it via the commands in the commandBuffers
			, waitStages 
			, renderCommandBuffers
			, isImageRenderedSemaphore // Raise when finished executing the commands
		}, isRenderCommandBufferExecutedFence); // Raise when finished executing the commands
	
		const auto presentResult = queue.presentKHR(vk::PresentInfoKHR{
			isImageRenderedSemaphore
			, swapchain
			, imageIndex
		});
		if (presentResult != vk::Result::eSuccess) throw std::runtime_error{"Failed to present image."};
	}
	void postRenderLoop(const ApplicationInfo& applicationInfo)
	{
		APPLICATION_INFO_BINDINGS

		device.destroyPipeline(computePipeline);
		device.destroyPipelineLayout(computePipelineLayout);
		device.destroyShaderModule(volumeShaderModule);
		device.destroyDescriptorSetLayout(descriptorSetLayout);
		device.destroySampler(volumeImageSampler);
		device.destroyDescriptorPool(descriptorPool);
		device.freeMemory(stagingBufferMemory); device.destroyBuffer(stagingBuffer);
		device.destroyImageView(volumeImageView); device.destroyImageView(raycastedImageView);
		device.freeMemory(volumeImageMemory); device.destroyImage(volumeImage);
		device.freeMemory(raycastedImageMemory); device.destroyImage(raycastedImage);
	}
}

int main()
{
	VulkanApplication application;
	const auto runInfo = RunInfo{
		{}
		, {}
		, vk::ImageUsageFlagBits::eTransferDst
		, preRenderLoop
		, renderFrame
		, postRenderLoop
		, "Volume Rendering"
	};
	application.run(runInfo);

	return EXIT_SUCCESS;
}

