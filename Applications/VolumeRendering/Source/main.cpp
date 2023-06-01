#include "VulkanApplication.h"
#include "vulkan/vulkan.hpp" // Do not place this header above the VulkanApplication.h
#include "Shader.h"
#include "Allocation.h"
#define IMGUI_DEFINE_MATH_OPERATORS
#include "imgui.h"
#include "imgui_impl_vulkan.h"
#include "glm/mat4x4.hpp"
#include "glm/gtx/transform.hpp"
#include <thread> // In case of wanting more workers
#include <cstdint> // Needed for uint32_t
#include <limits> // Needed for std::numeric_limits
#include <cmath>
#include <fstream>
#include <filesystem>
#include <vector>
#include <cassert>

#define APPLICATION_INFO_BINDINGS const auto& [window, instance, surface, physicalDevice, device, queue, queueFamilies, swapchain, surfaceFormat, surfaceExtent, commandBuffer] = applicationInfo;

namespace 
{
	// Application
	struct ViewPyramid
	{
		ViewPyramid(
			const glm::vec3& location_
			, float height_
			, const glm::vec3& side_
			, const glm::vec3& up_
			, const glm::vec3& forward_
		) : location{location_}
			, height{height_}
			, side{side_}
			, up{up_}
			, forward{forward_}
		{}

		glm::vec3 location; // Top of the pyramid (camera, the apex)
		float height; // Length from the top to the base of the pyramid (image grid)
		// Camera orthonormal basis
		glm::vec3 side;
		glm::vec3 up;
		glm::vec3 forward;
	};
	using Intensity = float; // Can't do short because sampler3D will always return a vec of floats
	constexpr auto windowExtent = glm::u32vec2{1280, 720}; // (800, 800), (1280, 720)
	const auto PI = std::acos(-1.0f);
	constexpr auto NUM_SLIDES = 113; // Ratio is 1:1:2, 2 unit of depth
	constexpr auto SLIDE_HEIGHT = 256;
	constexpr auto SLIDE_WIDTH = 256;
	constexpr auto NUM_INTENSITIES = NUM_SLIDES * SLIDE_HEIGHT * SLIDE_WIDTH;
	constexpr auto TOTAL_SCAN_BYTES = NUM_INTENSITIES * sizeof(Intensity);
	// z-y-x order, contains all intensity values of each slide images. Tightly packed vector instead of array because NUM_INTENSITIES is large, occupy the heap instead
	auto intensities = Resource{std::vector<Intensity>(NUM_INTENSITIES), toVulkanFormat<Intensity>()}; // The actual data used for sampling
	auto shaderMap = std::unordered_map<std::string, Shader>{};
	auto viewPyramid = ViewPyramid{
		glm::vec3{0.0f, 0.0f, 0.0f}
		, 0.0f
		, glm::vec3{1.0f, 0.0f, 0.0f}
		, glm::vec3{0.0f, 1.0f, 0.0f}
		, glm::vec3{0.0f, 0.0f, -1.0f}
	};
	vk::Image volumeImage, transferImage, raycastedImage, presentedImguiImage;
	vk::DeviceMemory volumeImageMemory, volumeImageStagingBufferMemory, transferImageMemory, transferImageStagingBufferMemory, raycastedImageMemory, presentedImguiImageMemory;
	vk::ImageView volumeImageView, transferImageView, raycastedImageView, presentedImguiImageView;
	vk::Buffer volumeImageStagingBuffer, transferImageStagingBuffer;
	vk::Sampler sampler;
	vk::DescriptorSetLayout descriptorSetLayout;
	vk::DescriptorPool descriptorPool;
	std::vector<vk::DescriptorSet> descriptorSets; vk::DescriptorSet imguiDescriptorSet;
	vk::ShaderModule volumeShaderModule;
	vk::PipelineLayout computePipelineLayout;
	vk::Pipeline computePipeline;

	// ImGui
	struct ControlPoint
	{
		ImVec2 position; // With respect to the histogram child window cursor, are always whole numbers
		ImColor color; // With alpha
	};
	constexpr auto imguiWindowExtent = ImVec2{500.0f, windowExtent.y}; // The height/width of the histogram/color spectrum
	auto histogram = std::vector<float>(100); // For imgui representation of the intensities histogram. The size of the histogram is the number of samples.
	auto controlPoints = std::vector<ControlPoint>{}; // Clicked control points, the position is relative to the histogram window cursor
	// Num transfer pixel is equal to the width of the color spectrum
	// Default color red when imgui histogram control points is empty
	auto transferFunction = Resource{std::vector<glm::vec4>(imguiWindowExtent.x, glm::vec4{0.0f, 0.0f, 0.0f, 0.0f}), toVulkanFormat<glm::vec4>()}; // The width of imgui window contains the number of pixel for the transfer function
	auto horizontalAngle = 0.0f;
	auto verticalAngle = 0.0f;

	void loadVolumeData()
	{
		auto dataIndex = 0;
		for (auto slideIndex = 1; slideIndex <= NUM_SLIDES; slideIndex++)
		{
			const auto ctPath = std::filesystem::path{VOLUME_DATA"/CThead." + std::to_string(slideIndex)};
			auto ctFile = std::ifstream{ctPath, std::ios_base::binary};
			if (!ctFile) throw std::runtime_error{"Can't open file at " + ctPath.string()};

			// Volume data is of type short, mentioned in the Standford document
			for (auto intensity = uint16_t{0}; ctFile.read(reinterpret_cast<char*>(&intensity), sizeof(intensity)); dataIndex++)
			{
				// Swap byte order if running on little-endian system
				#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
					data = (data >> 8) | (data << 8);
				#endif
				intensities.data[dataIndex] = intensity;
			}
		}
		// Normalize the data to [0, 1]
		const auto iterMax = std::ranges::max_element(intensities.data);
		const auto maxIntensity = *iterMax;
		const auto normalize = [&](Intensity& intensity){ return intensity /= maxIntensity; };
		std::ranges::for_each(intensities.data, normalize);
	}
	void prepareHistogramVolumeData()
	{
		auto sortedContainer = intensities.data;
		auto countMax = size_t{0};
		const auto stepSize = 1.0f / histogram.size();
		std::ranges::sort(sortedContainer, [](Intensity a, Intensity b){return a < b;}); // Ascending order
		for (int i = 0; i < histogram.size(); i++)
		{
			const auto iterLower = std::ranges::lower_bound(sortedContainer, stepSize * i); // Inclusiveness
			const auto iterUpper = std::ranges::upper_bound(sortedContainer, stepSize * (i + 0.9)); // 0.9 for Exclusiveness
			const auto count = static_cast<float>(std::distance(iterLower, iterUpper)); // Count the values within the bound
			histogram[i] = count;
			if (count > countMax) countMax = count;
		}
		for (auto& count : histogram) count /= countMax; // Normalize the count so we use it for scaling later on
	}
	void initComputePipeline(const ApplicationInfo& applicationInfo)
	{
		APPLICATION_INFO_BINDINGS
		const auto queueFamiliesIndex = std::views::elements<0>(queueFamilies) | std::ranges::to<std::vector>();
		// Local functions
		const auto setupTransferImageDeviceMemory = [&]()
		{
			// Reserve device memory for the transfer image
			transferImage = device.createImage(vk::ImageCreateInfo{
				{}
				, vk::ImageType::e1D
				, transferFunction.format
				, vk::Extent3D{static_cast<uint32_t>(transferFunction.data.size()), 1, 1}
				, 1
				, 1
				, vk::SampleCountFlagBits::e1
				, vk::ImageTiling::eOptimal
				, vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst
				, vk::SharingMode::eExclusive
				, queueFamiliesIndex
			});
			transferImageMemory = allocateMemory(device, physicalDevice, device.getImageMemoryRequirements(transferImage), false); // TODO: WHY DOESN'T THIS ALLOW HOST-VISIBLE MEMORY?
			device.bindImageMemory(transferImage, transferImageMemory, 0); // Associate the image handle to the memory handle
		};
		const auto setupTransferImageStagingBuffer = [&]()
		{
			// Reserve device memory for staging buffer
			transferImageStagingBuffer = device.createBuffer(vk::BufferCreateInfo{
				{}
				, transferFunction.data.size() * sizeof(glm::vec4)
				, vk::BufferUsageFlagBits::eTransferSrc
				, vk::SharingMode::eExclusive
				, queueFamiliesIndex
			});
			const auto stagingBufferMemoryRequirement = device.getBufferMemoryRequirements(transferImageStagingBuffer);
			transferImageStagingBufferMemory = allocateMemory(device, physicalDevice, stagingBufferMemoryRequirement, true);
			device.bindBufferMemory(transferImageStagingBuffer, transferImageStagingBufferMemory, 0);
		};
		const auto setupVolumeImageDeviceMemory = [&]()
		{
			volumeImage = device.createImage(vk::ImageCreateInfo{
				{}
				, vk::ImageType::e3D
				, intensities.format
				, {SLIDE_WIDTH, SLIDE_HEIGHT, NUM_SLIDES}
				, 1
				, 1
				, vk::SampleCountFlagBits::e1
				, vk::ImageTiling::eOptimal
				, vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst // Volumn data is transfered from a staging buffer transferSrc
				, vk::SharingMode::eExclusive
				, queueFamiliesIndex
			});
			volumeImageMemory = allocateMemory(device, physicalDevice, device.getImageMemoryRequirements(volumeImage), false);
			device.bindImageMemory(volumeImage, volumeImageMemory, 0);
		};
		const auto setupVolumeImageStagingBuffer = [&]()
		{
			// Reserve device memory for staging buffer
			volumeImageStagingBuffer = device.createBuffer(vk::BufferCreateInfo{
				{}
				, TOTAL_SCAN_BYTES
				, vk::BufferUsageFlagBits::eTransferSrc // Store and transfer volume data to volume image stored on the device memory via a copy command
				, vk::SharingMode::eExclusive
				, queueFamiliesIndex
			});
			const auto stagingBufferMemoryRequirement = device.getBufferMemoryRequirements(volumeImageStagingBuffer);
			volumeImageStagingBufferMemory = allocateMemory(device, physicalDevice, stagingBufferMemoryRequirement, true);
			device.bindBufferMemory(volumeImageStagingBuffer, volumeImageStagingBufferMemory, 0);

			// Upload volume data to the staging buffer, we will transfer this staging buffer data over to the volume image later
			void* memory = device.mapMemory(volumeImageStagingBufferMemory, 0, stagingBufferMemoryRequirement.size);
			std::memcpy(memory, intensities.data.data(), TOTAL_SCAN_BYTES);
			device.unmapMemory(volumeImageStagingBufferMemory);
		};
		const auto setupRaycastedImageDeviceMemory = [&]()
		{
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
			raycastedImageMemory = allocateMemory(device, physicalDevice, device.getImageMemoryRequirements(raycastedImage), false);
			device.bindImageMemory(raycastedImage, raycastedImageMemory, 0);
		};
		const auto setupRaycastedImageImguiDeviceMemory = [&]()
		{
			// Reserve device memory for the raycasted image
			presentedImguiImage = device.createImage(vk::ImageCreateInfo{
				{}
				, vk::ImageType::e2D
				, surfaceFormat
				, vk::Extent3D{surfaceExtent, 1}
				, 1 // The only mip level for this image
				, 1 // Single layer, no stereo-rendering
				, vk::SampleCountFlagBits::e1 // One sample, no multisampling
				, vk::ImageTiling::eOptimal
				, vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst //| vk::ImageUsageFlagBits::eTransientAttachment // Is sampled by ImGui shaders
				, vk::SharingMode::eExclusive
				, queueFamiliesIndex
			});
			presentedImguiImageMemory = allocateMemory(device, physicalDevice, device.getImageMemoryRequirements(presentedImguiImage), false);
			device.bindImageMemory(presentedImguiImage, presentedImguiImageMemory, 0);
		};

		const auto initDescriptorPool = [&]()
		{
			// Describe the size of each used descriptor type and the number of descriptor sets (To check against the association/create layout step)
			const auto descriptorPoolSizes = std::vector<vk::DescriptorPoolSize>{
				{vk::DescriptorType::eCombinedImageSampler, 2} // Volume image and transfer image
				, {vk::DescriptorType::eStorageImage, 1} // Raycasted image
			};
			descriptorPool = device.createDescriptorPool(vk::DescriptorPoolCreateInfo{
				{}
				, 1
				, descriptorPoolSizes
			});
		};
		const auto initDescriptorSets = [&]()
		{
			// Describe the binding numbers of the descriptors (To check against the association step)
			sampler = device.createSampler(vk::SamplerCreateInfo{
				{}
				//, vk::Filter::eNearest // TODO: try linear
				, vk::Filter::eLinear
				//, vk::Filter::eNearest // TODO: try linear
				, vk::Filter::eLinear
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
			const auto layoutBindings = {
				// Volume image
				vk::DescriptorSetLayoutBinding{
					0
					, vk::DescriptorType::eCombinedImageSampler
					, vk::ShaderStageFlagBits::eCompute
					, sampler
				}
				// Raycasted image
				, vk::DescriptorSetLayoutBinding{
					1
					, vk::DescriptorType::eStorageImage
					, vk::ShaderStageFlagBits::eCompute
					, sampler
				}
				// Transfer texel buffer
				, vk::DescriptorSetLayoutBinding{
					2
					, vk::DescriptorType::eCombinedImageSampler
					, vk::ShaderStageFlagBits::eCompute
					, sampler
				}
			};
			descriptorSetLayout  = device.createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo{{}, layoutBindings});
			descriptorSets = device.allocateDescriptorSets(vk::DescriptorSetAllocateInfo{descriptorPool, descriptorSetLayout});
		};

		// Format check for supporting operations
		checkFormatFeatures(physicalDevice, surfaceFormat, vk::FormatFeatureFlagBits::eStorageImage);
		checkFormatFeatures(physicalDevice, surfaceFormat, vk::FormatFeatureFlagBits::eTransferDst);
		checkFormatFeatures(physicalDevice, surfaceFormat, vk::FormatFeatureFlagBits::eTransferSrc);
		checkFormatFeatures(physicalDevice, intensities.format, vk::FormatFeatureFlagBits::eSampledImage);
		checkFormatFeatures(physicalDevice, intensities.format, vk::FormatFeatureFlagBits::eTransferDst);
		checkFormatFeatures(physicalDevice, transferFunction.format, vk::FormatFeatureFlagBits::eSampledImage);
		checkFormatFeatures(physicalDevice, transferFunction.format, vk::FormatFeatureFlagBits::eTransferDst);

		// Setups
		setupTransferImageDeviceMemory();
		setupTransferImageStagingBuffer();
		setupVolumeImageDeviceMemory();
		setupVolumeImageStagingBuffer();
		setupRaycastedImageDeviceMemory();
		setupRaycastedImageImguiDeviceMemory();

		// Inits
		initDescriptorPool();
		initDescriptorSets();

		// Associate the binding numbers to the descriptor sets
		volumeImageView = device.createImageView(vk::ImageViewCreateInfo{
			{}
			, volumeImage
			, vk::ImageViewType::e3D
			, intensities.format
			, {vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity}
			, vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
		});
		auto descriptorImageInfo = vk::DescriptorImageInfo{
			sampler
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

		raycastedImageView = device.createImageView(vk::ImageViewCreateInfo{
			{}
			, raycastedImage
			, vk::ImageViewType::e2D
			, surfaceFormat
			, {vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity}
			, vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
		});
		descriptorImageInfo = vk::DescriptorImageInfo{
			sampler
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

		transferImageView = device.createImageView(vk::ImageViewCreateInfo{
			{}
			, transferImage
			, vk::ImageViewType::e1D
			, transferFunction.format
			, {vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity}
			, vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
		});
		descriptorImageInfo = vk::DescriptorImageInfo{
			sampler
			, transferImageView 
			, vk::ImageLayout::eShaderReadOnlyOptimal
		};
		device.updateDescriptorSets(vk::WriteDescriptorSet{
			descriptorSets.front()
			, 2
			, 0
			, vk::DescriptorType::eCombinedImageSampler
			, descriptorImageInfo
			, {}
			, {}
		}, {});

		presentedImguiImageView = device.createImageView(vk::ImageViewCreateInfo{
			{}
			, presentedImguiImage
			, vk::ImageViewType::e2D
			, surfaceFormat
			, {vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity}
			, vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
		});

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
	void preRenderLoop(const ApplicationInfo& applicationInfo)
	{
		APPLICATION_INFO_BINDINGS

		loadVolumeData();
		prepareHistogramVolumeData();
		validateShaders(shaderMap);
		initComputePipeline(applicationInfo);
		// Create (via the imgui descriptor pool in the Vulkan application),
		// and update the imgui descriptor set (used by the imgui fragment
		// shader, as an input attachment) with the image view to display the
		// rendered raycasted image
		imguiDescriptorSet = ImGui_ImplVulkan_AddTexture(sampler, presentedImguiImageView, static_cast<VkImageLayout>(vk::ImageLayout::eShaderReadOnlyOptimal));
		submitCommandBufferOnceSynced(device, queue, commandBuffer, [&](const vk::CommandBuffer& commandBuffer){
			// Volume image layout: undefined -> transferDstOptimal, which is the expected layout when using the copy command
			commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, vk::ImageMemoryBarrier{
				vk::AccessFlagBits::eNone
				, vk::AccessFlagBits::eTransferWrite
				, vk::ImageLayout::eUndefined
				, vk::ImageLayout::eTransferDstOptimal
				, VK_QUEUE_FAMILY_IGNORED // Same queue family, don't transfer the queue ownership
				, VK_QUEUE_FAMILY_IGNORED
				, volumeImage
				, vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
			});
			// Copy the volume data from the staging buffer to the volume image used by the descriptor
			commandBuffer.copyBufferToImage(volumeImageStagingBuffer, volumeImage, vk::ImageLayout::eTransferDstOptimal, vk::BufferImageCopy{
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
			commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eBottomOfPipe, {}, {}, {}, {vk::ImageMemoryBarrier{
				vk::AccessFlagBits::eTransferWrite
				, vk::AccessFlagBits::eNone
				, vk::ImageLayout::eTransferDstOptimal
				, vk::ImageLayout::eShaderReadOnlyOptimal
				, VK_QUEUE_FAMILY_IGNORED // Same queue family, don't transfer the queue ownership
				, VK_QUEUE_FAMILY_IGNORED
				, volumeImage
				, vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
			}});
			// Swapchain images layout: undefined -> presentSrcKHR. Layout transition of the swapchain images to presentSrcKHR (PRESENTABLE IMAGE) in order to acquire the image index during the rendering loop without causing error
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
		});
	}

	void updateTransferFunction(const std::vector<ControlPoint>& controlPoints, std::vector<glm::vec4>& transferFunction)
	{
		if (controlPoints.empty()) return; // First frame, the 2 default control points aren't pushed yet. The transferFunction will have its elements assigned with glm::vec4{0.0f, 0.0f, 0.0f, 0.0f}
		for (const auto& [i, controlPoint] : std::views::enumerate(controlPoints))
		{
			if (i == 0) continue;
			const auto colorDirection = subtract(controlPoints[i].color, controlPoints[i - 1].color); // Scaled color vector going from the previous' to this control point's color
			const auto totalPixels = controlPoints[i].position.x - controlPoints[i - 1].position.x; // The number of pixels that will be assigned by this and the previous control points
			for (const auto j : std::views::iota(0, totalPixels)) // Inclusive
			{
				const auto interpolatedColor = add(controlPoints[i].color, scale(colorDirection, static_cast<float>(j) / totalPixels)); // Perform linear interpolation
				transferFunction[controlPoints[i - 1].position.x + j] = glm::vec4{
					interpolatedColor.x
					, interpolatedColor.y
					, interpolatedColor.z
					, 1.0f - interpolatedColor.w // Interpolated color's w (alpha) goes to 0 means hitting the ceiling of the histogram window -> increasing alpha (= 1 - w)
				};
			}
		}
	}
	void updateTransferImage(const ApplicationInfo& applicationInfo)
	{
		APPLICATION_INFO_BINDINGS

		updateTransferFunction(controlPoints, transferFunction.data);

		// Map the memory handle to the actual device memory (host visible, specified in the allocateMemory()) to upload the data
		void* memory = device.mapMemory(transferImageStagingBufferMemory, 0, device.getBufferMemoryRequirements(transferImageStagingBuffer).size);
		std::memcpy(memory, transferFunction.data.data(), transferFunction.data.size() * sizeof(glm::vec4)); // The size should be shared with the setupTransferTex... with a Resource
		device.unmapMemory(transferImageStagingBufferMemory);

		// Transfer image layout: undefined -> transferDstOptimal
		commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, vk::ImageMemoryBarrier{
			vk::AccessFlagBits::eNone
			, vk::AccessFlagBits::eTransferWrite
			, vk::ImageLayout::eUndefined // Default and clear the previous image, BUT DOESN't have to if no new control points are added
			, vk::ImageLayout::eTransferDstOptimal
			, VK_QUEUE_FAMILY_IGNORED // Same queue family, don't transfer the queue ownership
			, VK_QUEUE_FAMILY_IGNORED
			, transferImage
			, vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
		});

		// Copy the volume data from the staging buffer to the volume image used by the descriptor
		commandBuffer.copyBufferToImage(transferImageStagingBuffer, transferImage, vk::ImageLayout::eTransferDstOptimal, vk::BufferImageCopy{
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
			, vk::Extent3D{static_cast<uint32_t>(transferFunction.data.size()), 1, 1}
		});

		// Transfer image layout: transferDstOptimal -> shaderReadOnlyOptimal
		commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader, {}, {}, {}, {vk::ImageMemoryBarrier{
			vk::AccessFlagBits::eTransferWrite
			, vk::AccessFlagBits::eShaderRead
			, vk::ImageLayout::eTransferDstOptimal
			, vk::ImageLayout::eShaderReadOnlyOptimal
			, VK_QUEUE_FAMILY_IGNORED // Same queue family, don't transfer the queue ownership
			, VK_QUEUE_FAMILY_IGNORED
			, transferImage
			, vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
		}});

	}
	// imgui 2 slider
	// the height of the pyramid == the distance of the top of the pyramaid
	// The location can't be modified, is implicitly modifed by the sliders
	// vertical rotation around the vr
	// horizontal rotation around the vr
	auto getViewPyramid(){
		//const auto pitch = glm::rotate(x * PI, glm::vec3{1.0f, 0.0f, 0.0f}); // Pitch
		//const auto yaw = glm::rotate(y * PI, glm::vec3{0.0f, 1.0f, 0.0f}); // Yaw
		//const auto roll = glm::rotate(z * PI, glm::vec3{0.0f, 0.0f, -1.0f}); // Roll
	}
	auto /*glm::mat4x4*/ getPixelTransform(){}
	void renderCommands(const ApplicationInfo& applicationInfo, uint32_t imageIndex, bool isFirstFrame)
	{
		APPLICATION_INFO_BINDINGS

		updateTransferImage(applicationInfo);

		// Transition layout of the image descriptors before compute pipeline
		// Raycasted image layout: undefined -> general, expected by the descriptor
		commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eComputeShader, {}, {}, {}, vk::ImageMemoryBarrier{
			vk::AccessFlagBits::eNone
			, vk::AccessFlagBits::eShaderWrite
			, vk::ImageLayout::eUndefined // Default & discard the previous contents of the raycastedImage
			, vk::ImageLayout::eGeneral
			, VK_QUEUE_FAMILY_IGNORED // Same queue family, don't transfer the queue ownership
			, VK_QUEUE_FAMILY_IGNORED
			, raycastedImage
			, vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
		});

		commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, computePipeline);
		commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, computePipelineLayout, 0, descriptorSets.front(), {});
		const auto numInvocationPerX = 10; // Also put this number in the compute shader
		const auto numInvocationPerY = 10; // Also put this number in the compute shader
		assert((windowExtent.x % numInvocationPerX) == 0 && (windowExtent.y % numInvocationPerY) == 0);
		commandBuffer.dispatch(windowExtent.x / numInvocationPerX, windowExtent.y / numInvocationPerY, 1); // NOTE: group size must be at least 1 for all x, y, z

		//const auto swapchainImage = device.getSwapchainImagesKHR(swapchain)[imageIndex];
		// TODO: what about the rendered image that imgui is going render ontop of ?
		// we give to imgui as an input attachment via descriptor, should be visible to imgui's fragment shader

		//// Barrier to sync writing to raycasted image via compute shader and copy it to swapchain image
		//// Raycasted image layout: general -> transferSrc, before copy command
		//// Swapchain image layout: undefined -> transferDst, before copy command
		//commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, {
		//	vk::ImageMemoryBarrier{
		//		vk::AccessFlagBits::eShaderWrite // Compute shader writes to raycasted image, vk::AccessFlagBits::eShaderWrite, only use this when the shader write to the memory
		//		, vk::AccessFlagBits::eTransferRead // Wait until raycasted image is finished written to then copy
		//		, vk::ImageLayout::eGeneral
		//		, vk::ImageLayout::eTransferSrcOptimal
		//		, VK_QUEUE_FAMILY_IGNORED // Same queue family, don't transfer the queue ownership
		//		, VK_QUEUE_FAMILY_IGNORED // Same queue family, don't transfer the queue ownership
		//		, raycastedImage
		//		, vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
		//	}
		//	, vk::ImageMemoryBarrier{
		//		vk::AccessFlagBits::eNone
		//		, vk::AccessFlagBits::eTransferWrite // Wait until raycasted image is finished written to then copy
		//		, vk::ImageLayout::eUndefined // Default & discard the previous contents of the swapchainImage
		//		, vk::ImageLayout::eTransferDstOptimal
		//		, VK_QUEUE_FAMILY_IGNORED // Same queue family, don't transfer the queue ownership
		//		, VK_QUEUE_FAMILY_IGNORED // Same queue family, don't transfer the queue ownership
		//		, swapchainImage
		//		, vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
		//	}
		//});

		//// Copy data from the rendered raycasted image via the compute shader to the current swapchain image
		//commandBuffer.copyImage(raycastedImage, vk::ImageLayout::eTransferSrcOptimal, swapchainImage, vk::ImageLayout::eTransferDstOptimal, vk::ImageCopy{
		//	vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, 0, 0, 1}
		//	, {0, 0, 0}
		//	, vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, 0, 0, 1}
		//	, {0, 0, 0}
		//	, vk::Extent3D{surfaceExtent, 1}
		//});

		// Swapchain image layout: transferDst -> colorAttachment, for UI imgui renderpass
		//commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eBottomOfPipe, {}, {}, {}, vk::ImageMemoryBarrier{
		//	vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite
		//	, vk::AccessFlagBits::eNone
		//	, vk::ImageLayout::eTransferDstOptimal
		//	, vk::ImageLayout::eColorAttachmentOptimal
		//	, VK_QUEUE_FAMILY_IGNORED
		//	, VK_QUEUE_FAMILY_IGNORED
		//	, device.getSwapchainImagesKHR(swapchain)[imageIndex]
		//	, vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}

		// don't copy the raycasted image to the swapchain since we don't want to render it directly to the viewport but through an imgui window
		commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, vk::ImageMemoryBarrier{
			vk::AccessFlagBits::eShaderWrite
			, vk::AccessFlagBits::eTransferRead
			, vk::ImageLayout::eGeneral
			, vk::ImageLayout::eTransferSrcOptimal
			, VK_QUEUE_FAMILY_IGNORED
			, VK_QUEUE_FAMILY_IGNORED
			, raycastedImage
			, vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
		});

		commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eBottomOfPipe, vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, vk::ImageMemoryBarrier{
			vk::AccessFlagBits::eNone
			, vk::AccessFlagBits::eTransferWrite
			, vk::ImageLayout::eUndefined
			, vk::ImageLayout::eTransferDstOptimal
			, VK_QUEUE_FAMILY_IGNORED
			, VK_QUEUE_FAMILY_IGNORED
			, presentedImguiImage
			, vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
		});


		// What about if we just pass the descriptor set with 3 descriptor to imgui?
		commandBuffer.copyImage(raycastedImage, vk::ImageLayout::eTransferSrcOptimal, presentedImguiImage, vk::ImageLayout::eTransferDstOptimal, vk::ImageCopy{
			vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, 0, 0, 1}
			, {0, 0, 0}
			, vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, 0, 0, 1}
			, {0, 0, 0}
			, vk::Extent3D{surfaceExtent, 1}
		});

		// just in case when used via ocmmands specified in imgui's command buffer?
		// TODO: test disable this since we already have the event barrier? And because the pipeline barrier only work within a command buffer?
		commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eBottomOfPipe, {}, {}, {}, vk::ImageMemoryBarrier{
			vk::AccessFlagBits::eTransferWrite
			, vk::AccessFlagBits::eNone
			, vk::ImageLayout::eTransferDstOptimal
			, vk::ImageLayout::eShaderReadOnlyOptimal
			, VK_QUEUE_FAMILY_IGNORED
			, VK_QUEUE_FAMILY_IGNORED
			, presentedImguiImage
			, vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
		});

		// Swapchain image layout: undefined -> colorAttachment, for UI imgui renderpass
		commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eBottomOfPipe, {}, {}, {}, vk::ImageMemoryBarrier{
			vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite
			, vk::AccessFlagBits::eNone // Discard previous content, the imgui renderpass will produce an image from its shader + the raycasted imgui descriptor
			, vk::ImageLayout::eUndefined
			, vk::ImageLayout::eColorAttachmentOptimal
			, VK_QUEUE_FAMILY_IGNORED
			, VK_QUEUE_FAMILY_IGNORED
			, device.getSwapchainImagesKHR(swapchain)[imageIndex]
			, vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
		});
	}

	// ImGui, root & child windows
	void childHistogramCommands(const ImVec2& childExtent, const ImColor& backgroundColor)
	{
		// Statics
		static auto hitControlPoint = controlPoints.end(); // Currently clicked control point
		static auto isDraggingControlPoint = false;

		// Default values and helper functions
		const auto defaultControlPointColor = ImColor{0.0f, 0.0f, 0.0f, 1.0f};
		const auto controlPointRadius = 5.0f;
		const auto getAlpha = [](const ImVec2& controlPointPosition, const ImVec2& childWindowExtent)
		{
			// The controlPointPosition is with respect to the child window cursor
			return 1.0f - (controlPointPosition.y / childWindowExtent.y); // Control point's y = 0 (hit the child window's ceiling) means max opacity = 1
		};
		const auto getHitControlPoint = [&](const ImVec2& cursor, const float tolerance = 10.0f)
		{
			// Return an optional index to a hit control point. Cursor is the cursor of the window in which the user clicks
			return std::ranges::find_if(controlPoints, [&](const ControlPoint& controlPoint)
			{
				// Add the cursor and the control point position to get to the screen space position instead of the control point being relative to the cursor
				return length(subtract(add(cursor, controlPoint.position), ImGui::GetMousePos())) <= controlPointRadius + tolerance;
			});
		};

		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2{0, 0}); // Needed to avoid padding between the child window and the invisible button/the clip rect, tested with a visible button
		ImGui::PushStyleColor(ImGuiCol_ChildBg, static_cast<ImVec4>(backgroundColor));
		if (ImGui::BeginChild("Histogram Alpha", childExtent, true))
		{
			//TODO: ImGui::GetWindowSize() after this instead of getChildExtent()?
			const auto childWindowCursor = ImGui::GetCursorScreenPos(); // Must be placed above the visible button because we want the cursor of the current child window, not the button
			const auto drawList = ImGui::GetWindowDrawList();

			// Two default control points that aren't using the default control point color
			if (controlPoints.empty())
			{
				// TODO: if the extend is maxX, maxY. Must subtract it by 1 because [0, max) exclusive
				const auto leftMostControlPoint = unnormalizeCoordinate(ImVec2{0.0f, 0.5f}, childExtent);
				controlPoints.push_back(ControlPoint{
					leftMostControlPoint
					, ImColor{1.0f, 0.0f, 0.0f, getAlpha(leftMostControlPoint, childExtent)
					}
				});
				const auto rightMostControlPoint = unnormalizeCoordinate(ImVec2{1.0f, 0.5f}, childExtent);
				controlPoints.push_back(ControlPoint{
					rightMostControlPoint
					, ImColor{0.0f, 0.0f, 1.0f, getAlpha(rightMostControlPoint, childExtent)
					}
				});
			}
			assert(controlPoints.front().position.y >= 0 && controlPoints.back().position.y >= 0); // Sanity check

			// Mouse position capture area, push back any captured position (control point) for drawing
			ImGui::InvisibleButton("Input position capture", childExtent, ImGuiMouseButton_Left | ImGuiMouseButton_Right);
			if (ImGui::IsMouseDown(ImGuiMouseButton_Left))
			{
				if (ImGui::IsItemClicked(ImGuiMouseButton_Left)) // Hovering over doesn't count as clicked, will automatically goes to the else if if dragging a control point
				{
					hitControlPoint = getHitControlPoint(childWindowCursor);
					if (hitControlPoint != controlPoints.end()) isDraggingControlPoint = true;
					else // Add new control point
					{
						const auto clickedPositionGLFW = ImGui::GetMousePos(); // With respect to the glfw Window
						const auto clickedPositionInChild = subtract(clickedPositionGLFW, childWindowCursor); // with respect to the child window button
						controlPoints.push_back(ControlPoint{
							clickedPositionInChild
							, ImColor{
								defaultControlPointColor.Value.x
								, defaultControlPointColor.Value.y
								, defaultControlPointColor.Value.z
								, getAlpha(clickedPositionInChild, childExtent)
							}
						});
						std::ranges::sort(controlPoints, [](const ImVec2& a, const ImVec2& b) {return a.x < b.x;}, [](const ControlPoint& controlPoint) {return controlPoint.position;}); // Only sort the control points in an ascending order based on the x position
					}
				}
				else if (isDraggingControlPoint) // Only changing the y axis
				{
					const auto clickedPositionGLFW = ImGui::GetMousePos(); // With respect to the glfw Window
					auto clickedPositionInChild = subtract(clickedPositionGLFW, childWindowCursor);

					// Clipping, constraint the draggable area
					if (clickedPositionInChild.y < 0.0f) clickedPositionInChild.y = 0.0f;
					else if (clickedPositionInChild.y > childExtent.y - 1) clickedPositionInChild.y = childExtent.y - 1;
					// Restrict to only y-axis if dragging the control point at either ends
					if (hitControlPoint != controlPoints.begin() && hitControlPoint != controlPoints.end() - 1)
					{
						if (clickedPositionInChild.x < 0.0f) clickedPositionInChild.x = 0.0f;
						else if (clickedPositionInChild.x > childExtent.x - 1) clickedPositionInChild.x = childExtent.x - 1;
					}
					else clickedPositionInChild.x = hitControlPoint->position.x;
					hitControlPoint->position.y = clickedPositionInChild.y;
					hitControlPoint->position.x = clickedPositionInChild.x;
					// Update the alpha after shifted to a new position
					hitControlPoint->color.Value.w = getAlpha(clickedPositionInChild, childExtent);

					// Update the hitIndex location after sorted
					std::ranges::sort(controlPoints, [](const ImVec2& a, const ImVec2& b)
					{
						return a.x < b.x;
					}, [](const ControlPoint& controlPoint)
					{
						return controlPoint.position; // Only sort the control points in an ascending order based on the x position
					});
					hitControlPoint = std::ranges::find_if(controlPoints, [&](const ControlPoint& controlPoint){
						return controlPoint.position.x == clickedPositionInChild.x && controlPoint.position.y == clickedPositionInChild.y;
					});
				}
			}
			else isDraggingControlPoint = false;

			if (ImGui::IsItemClicked(ImGuiMouseButton_Right)) // If this is false then the popup won't be drawn, we need to seperate openpopup  and beginpopup instaed of placing both in this conditional statement
			{
				hitControlPoint = getHitControlPoint(childWindowCursor);
				if (hitControlPoint != controlPoints.end())
				{
					ImGui::OpenPopup("Input position capture");
				}
			}

			// Popups, need to be close to the outer loop to be invoked continuously to keep the popups stay opened
			if (ImGui::BeginPopup("Input position capture", ImGuiWindowFlags_NoMove)) // The menu items are drawn above the push clip rect, we don't need to deferred it later than tthe clip rect drawing stage
			{
				//if (ImGui::MenuItem("Change color", nullptr))
				//{
				//	ImGui::OpenPopup("Change color");
				//}
				// Casting the address of the color of the current hit control point to float[3]. Since ImColor is the same as float[4], it is valid to do this.
				ImGui::ColorPicker3("###", (float*)&(hitControlPoint->color), ImGuiColorEditFlags_PickerHueBar | ImGuiColorEditFlags_NoSidePreview | ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoAlpha);
				if (ImGui::MenuItem("Delete") )
				{
					auto iterControlPoint = hitControlPoint;
					//std::advance(iterControlPoint, hitControlPoint.value());
					// Can't delete the first and the final control points
					if (controlPoints.size() > 2
						&& iterControlPoint != controlPoints.begin()
						&& iterControlPoint != controlPoints.end() - 1) controlPoints.erase(iterControlPoint);
				}
				ImGui::EndPopup();
			}

			// Drawing area within the current child window
			drawList->PushClipRect(childWindowCursor, add(childWindowCursor, childExtent));

			// Draw histogram
			const auto stepSize = 1.0f / histogram.size();
			for (int i = 0; i < histogram.size(); i++)
			{
				const auto upperLeftExtentX = i * stepSize * childExtent.x;
				const auto upperLeftExtentY = childExtent.y * (1.0f - histogram[i]); // count == histogram[i] is normalized, we can use it to scale the extent.y. We do (1.0f - count) to force the scale 0.0 and 1.0 to be further/closert to the child top left origin

				const auto lowerRightExtentX = (i + 1) * stepSize * childExtent.x;
				const auto lowerRightExtentY = childExtent.y;

				const auto upperLeft = add(childWindowCursor, ImVec2{upperLeftExtentX, upperLeftExtentY});
				const auto lowerRight = add(childWindowCursor, ImVec2{lowerRightExtentX, lowerRightExtentY});
				drawList->AddRectFilled(upperLeft, lowerRight, ImColor{0.5f, 0.5f, 0.5f});
			}

			// Draw any connection lines. Draw connection lines first then draw the control points to make the control points ontop of the lines
			for (int i = 1; i < controlPoints.size(); i++)
			{
				drawList->AddLine(
					add(childWindowCursor, controlPoints[i - 1].position) // The control points are with respect to the child window cursor, we need to offset them to the ImGui screen position before using them
					, add(childWindowCursor, controlPoints[i].position)
					, ImColor{ 1.0f, 1.0f, 1.0f });
			}

			// Draw control points
			for (int i = 0; i < controlPoints.size(); i++)
			{
				drawList->AddCircleFilled(
					add(childWindowCursor, controlPoints[i].position)
					, controlPointRadius
					, ImColor{
						controlPoints[i].color.Value.x
						, controlPoints[i].color.Value.y
						, controlPoints[i].color.Value.z
						, 1.0f // We want to see the color, the alpha of this color is implied by the position of the control point in the window and the color spectrum
					}
				, 0);
			}

			drawList->PopClipRect();
		}
		ImGui::EndChild();
		ImGui::PopStyleColor();
		ImGui::PopStyleVar();
	}
	void childColorSpectrumCommands(const ImVec2& childExtent, const ImColor& background)
	{
		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2{0, 0}); // Needed to avoid padding between the child window and the clip rect, tested with a visible button
		ImGui::PushStyleColor(ImGuiCol_ChildBg, static_cast<ImVec4>(background));  // Set a background color
		if (ImGui::BeginChild("Color spectrum", childExtent, true, ImGuiWindowFlags_NoMove))
		{
			const auto drawList = ImGui::GetWindowDrawList();
			const auto childWindowCursor = ImGui::GetCursorScreenPos();

			// Draw each of the color spectrum rectangles per 2 control points
			drawList->PushClipRect(childWindowCursor, add(childWindowCursor, childExtent));

			for (size_t i = 1; i < controlPoints.size(); i++)
			{
				const auto upperLeftPosition = ImVec2{childWindowCursor.x + controlPoints[i - 1].position.x, childWindowCursor.y};
				const auto leftColor = controlPoints[i - 1].color; // Including alpha

				const auto lowerRightPosition = ImVec2{childWindowCursor.x + controlPoints[i].position.x, childWindowCursor.y + childExtent.y}; // Adding y of the child extent to reach to the floor of the current child window
				const auto rightColor = controlPoints[i].color;  // Including alpha

				drawList->AddRectFilledMultiColor(
					upperLeftPosition
					, lowerRightPosition
					, leftColor // Upper left
					, rightColor // Upper right
					, rightColor // Lower right
					, leftColor // Lower left
				);
			}

			drawList->PopClipRect();

		}
		ImGui::EndChild();
		ImGui::PopStyleColor();
		ImGui::PopStyleVar();
	}
	void rootCommands()
	{
		//ImGui::ShowDemoWindow();
		//const auto& io = ImGui::GetIO();

		ImGui::SetNextWindowPos(ImVec2{0, 0}, ImGuiCond_Always);
		ImGui::SetNextWindowSize(imguiWindowExtent, ImGuiCond_Always);
		if (ImGui::Begin("Transfer function editor", 0, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse)) // Create a window. No resizing because this is mess up the control point positions, temporary doing this for now.
		{
			//ImGui::SetCursorScreenPos(ImVec2{0, 0});
			ImGui::SliderAngle("Rotate Horizontal", &horizontalAngle, -180.0f, 180.0f);
			ImGui::SliderAngle("Rotate Vertical", &verticalAngle, -90.0f, 90.0f);

			const auto backgroundColor = ImColor{0.5f, 0.5f, 0.5f, 0.5f};
			childHistogramCommands(ImVec2{
				ImGui::GetContentRegionAvail().x
				, (11.0f / 12.0f * ImGui::GetContentRegionAvail().y)// - (ImGui::GetStyle().FramePadding.y / 2.0f) // Frame padding divided by 2 because we have 2 child windows
			}, backgroundColor);
			childColorSpectrumCommands(ImVec2{
				// The remaining content region
				ImGui::GetContentRegionAvail().x
				, ImGui::GetContentRegionAvail().y
			}, backgroundColor);

			ImGui::SetNextWindowPos(ImVec2{imguiWindowExtent.x, 0}, ImGuiCond_Always); // Add one extra pixel
			ImGui::SetNextWindowSize(ImVec2{windowExtent.x - imguiWindowExtent.x - 10, windowExtent.y}, ImGuiCond_Always);
			if (ImGui::Begin("Rendered image", 0, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse))
			{
				ImGui::Image(imguiDescriptorSet, ImGui::GetContentRegionAvail());
				ImGui::End();
			}
		}
		ImGui::End();
	}

	void postRenderLoop(const ApplicationInfo& applicationInfo)
	{
		APPLICATION_INFO_BINDINGS

		device.destroyPipeline(computePipeline);
		device.destroyPipelineLayout(computePipelineLayout);
		device.destroyShaderModule(volumeShaderModule);
		device.destroyDescriptorSetLayout(descriptorSetLayout);
		device.destroySampler(sampler);
		device.destroyDescriptorPool(descriptorPool);

		ImGui_ImplVulkan_RemoveTexture(imguiDescriptorSet);
		device.destroyImageView(presentedImguiImageView);
		device.freeMemory(presentedImguiImageMemory);
		device.destroyImage(presentedImguiImage);

		device.destroyImageView(volumeImageView);
		device.freeMemory(volumeImageMemory);
		device.destroyImage(volumeImage);
		device.freeMemory(volumeImageStagingBufferMemory);
		device.destroyBuffer(volumeImageStagingBuffer);

		device.destroyImageView(transferImageView);
		device.freeMemory(transferImageMemory);
		device.destroyImage(transferImage);
		device.freeMemory(transferImageStagingBufferMemory);
		device.destroyBuffer(transferImageStagingBuffer);

		device.destroyImageView(raycastedImageView);
		device.freeMemory(raycastedImageMemory);
		device.destroyImage(raycastedImage);
	}
}

int main()
{
	VulkanApplication application;
	const auto runInfo = RunInfo{
		{}
		, {}
		, preRenderLoop
		, renderCommands
		, rootCommands
		, postRenderLoop
		, "Volume Rendering"
		, windowExtent
	};
	application.run(runInfo);

	return EXIT_SUCCESS;
}

