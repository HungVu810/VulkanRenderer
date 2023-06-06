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

// constexpr, consteval
// create a appThread class that takes works and assigned with enum of the current work
// TODO: seperate this vuklan application into a framework to support different
// type of graphic program, ie volumn rendering, normal mesh renderng
// TODO: Split into multiple command buffer with 1 submission? Set event
// If application want multiple command buffer, it should create its own beside the provided one via ApplicationInfo
// TODO: port utilities function from os project over
// bezier curve efor the control points
// TODOD: imgui widgets for ray casting sample sizes, and controling the camera position
// TODO: control 3 rgb lines instead of control ponts?
// TODO: not all transferFunction .data is covered (ie, 489 out of 500, due to padding in the transfer window)
// TODO: checkot imgui tips for using math on imvec
//TODO: Don't share imgui renderpass with application renderpass, do this first beforer attempt the below
//TODO: Check if the runInfo already provdie a renderpass, grpahic pipeline, framebuffer,...?
//TODO: A variable to toggle imgui log messages
// TODO: Create a struct RenderFrame in RunInfo that accept a recording function and an optional imgui commands function, will be check against the USE_IMGUI var
// TODO: Remove isFirstFrame, this can be done in the preRenderLoop function
// TODO: Avoid the computation if the controlPoints doesn't change
// todo: totalPixels != imguiWindowExtents.width
// TODO: Half windows for visualization, half window for imgui histogram
//ImGuiIO& io = ImGui::GetIO(); (void)io;
//io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
//io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
//// Setup Dear ImGui style
//ImGui::StyleColorsDark();
////ImGui::StyleColorsLight();
// TODO: Reduce the number of pixel need to be rendered since we only use half of the glfw window
namespace 
{
	// Main application
	using Intensity = float; // Can't do short because sampler3D will always return a vec of floats
	constexpr auto windowExtent = glm::u32vec2{1280, 720}; // (800, 800), (1280, 720)
	constexpr auto NUM_SLIDES = 113; // Ratio is 1:1:2, 2 unit of depth
	constexpr auto SLIDE_HEIGHT = 256;
	constexpr auto SLIDE_WIDTH = 256;
	constexpr auto NUM_INTENSITIES = NUM_SLIDES * SLIDE_HEIGHT * SLIDE_WIDTH;
	constexpr auto TOTAL_SCAN_BYTES = NUM_INTENSITIES * sizeof(Intensity);
	// z-y-x order, contains all intensity values of each slide images. Tightly packed vector instead of array because NUM_INTENSITIES is large, occupy the heap instead
	auto intensities = Resource{std::vector<Intensity>(NUM_INTENSITIES), toVulkanFormat<Intensity>()}; // The actual data used for sampling
	auto shaderMap = std::unordered_map<std::string, Shader>{};
	vk::Image volumeImage, transferImage, raycastedImage, presentedImguiImage;
	vk::Buffer volumeImageStagingBuffer, transferImageStagingBuffer, viewPyramidStructBuffer, pixelToWorldSpaceTransformBuffer;
	vk::DeviceMemory volumeImageMemory, volumeImageStagingBufferMemory, transferImageMemory, transferImageStagingBufferMemory, raycastedImageMemory, presentedImguiImageMemory, viewPyramidStructBufferMemory, pixelToWorldSpaceTransformBufferMemory;
	vk::ImageView volumeImageView, transferImageView, raycastedImageView, presentedImguiImageView;
	vk::Sampler sampler;
	vk::DescriptorSetLayout descriptorSetLayout;
	vk::DescriptorPool descriptorPool;
	std::vector<vk::DescriptorSet> descriptorSets; vk::DescriptorSet imguiDescriptorSet;
	vk::ShaderModule volumeShaderModule;
	vk::PipelineLayout computePipelineLayout;
	vk::Pipeline computePipeline;

	// Camera, contains the length from the top to the base of the pyramid
	// (image grid). The translating location (top of the pyramid (camera, the
	// apex)), and an local orthonormal basis. The type are all glm::vec4 to
	// account the alignment via std140
	struct ViewPyramid
	{
		glm::vec4 height; // First component stores the height
		// Orthogonal basis
		glm::vec4 location; // mat[3]
		glm::vec4 side; // mat[0]
		glm::vec4 up; // mat[1]
		glm::vec4 forward; // mat[2]
	};
	auto viewPyramid = ViewPyramid{ // Keep as the default orientation, only modifed by setupViewPyramid() then readonly
		.height      = glm::vec4{0}
		, .location  = glm::vec4{0}
		, .side      = glm::vec4{1, 0, 0, 0}
		, .up        = glm::vec4{0, 1, 0, 0}
		, .forward   = glm::vec4{0, 0, -1, 0}
	};
	constexpr auto volumeGridCenter = glm::vec4{0, 0, -0.5f, 0}; // Extent == [0, 1]^3. Do not modify.
	auto horizontalRadian = 0.0f;
	auto verticalRadian = 0.0f;

	// ImGui
	struct ControlPoint
	{
		ImVec2 position; // With respect to the histogram child window cursor, are always whole numbers
		ImColor color; // With alpha
	};
	constexpr auto imguiWindowExtent = ImVec2{500.0f, windowExtent.y}; // The height/width of the histogram/color spectrum
	auto histogram = std::vector<float>(100); // For imgui representation of the intensities histogram. The size of the histogram is the number of samples.
	auto controlPoints = std::vector<ControlPoint>{}; // Clicked control points, the position is relative to the histogram window cursor
	auto transferFunction = Resource{std::vector<glm::vec4>(imguiWindowExtent.x, glm::vec4{0.0f, 0.0f, 0.0f, 0.0f}), toVulkanFormat<glm::vec4>()}; // The width of imgui window contains the number of pixel for the transfer function

	void loadVolumeData()
	{
		auto dataIndex = 0;
		for (auto slideIndex = 1; slideIndex <= NUM_SLIDES; slideIndex++)
		{
			const auto ctPath = std::filesystem::path{VOLUME_DATA"/CThead." + std::to_string(slideIndex)};
			auto ctFile = std::ifstream{ctPath, std::ios_base::binary};
			if (!ctFile) throw std::runtime_error{"Can't open file at " + ctPath.string()};

			// Volume data is of type short, mentioned in the Stanford document
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
	void setupViewPyramid()
	{
		// The corners are the side of the cube if the image suddenly has edge corners due to perespective projection when adjusting the viewPyramidLocation/viewPyramidHeight
		// To rotate the viewPyramid forward, up and side 
		// Because the volume dimension is 1, rotating radian need to be small to compensate
		auto viewPyramidBasis = glm::mat4{
			viewPyramid.side
			, viewPyramid.up
			, viewPyramid.forward
			, glm::vec4{0, 0, 0, 1}
		};
		viewPyramidBasis =
			glm::translate(glm::vec3{0, -5.0f, -0.5f}) // Translate to view the bottom of the volume grid, where the CT face is
			* glm::rotate(glm::radians(90.0f), glm::vec3{1, 0, 0}) // Pitch up 90 degree
			* viewPyramidBasis;
		viewPyramid.height = glm::vec4{4.5f, 0, 0, 0};
		viewPyramid.side = viewPyramidBasis[0];
		viewPyramid.up = viewPyramidBasis[1];
		viewPyramid.forward = viewPyramidBasis[2];
		viewPyramid.location = viewPyramidBasis[3];
	}
	void initComputePipeline(const ApplicationInfo& applicationInfo)
	{
		APPLICATION_INFO_BINDINGS
		const auto queueFamiliesIndex = std::views::elements<0>(queueFamilies) | std::ranges::to<std::vector>();
		// Local functions
		// TODO: inline all lambdas instead of sperating the definition and the usage
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
		const auto setupViewPyramidStructBuffer = [&]()
		{
			// Reserve device memory for staging buffer
			viewPyramidStructBuffer = device.createBuffer(vk::BufferCreateInfo{
				{}
				, sizeof(ViewPyramid)
				, vk::BufferUsageFlagBits::eUniformBuffer
				, vk::SharingMode::eExclusive
				, queueFamiliesIndex
			});
			const auto stagingBufferMemoryRequirement = device.getBufferMemoryRequirements(viewPyramidStructBuffer);
			viewPyramidStructBufferMemory = allocateMemory(device, physicalDevice, stagingBufferMemoryRequirement, true);
			device.bindBufferMemory(viewPyramidStructBuffer, viewPyramidStructBufferMemory, 0);
		};
		const auto setupPixelToWorldSpaceTransformBuffer = [&]()
		{
			// Reserve device memory for staging buffer
			pixelToWorldSpaceTransformBuffer = device.createBuffer(vk::BufferCreateInfo{
				{}
				, sizeof(glm::mat4)
				, vk::BufferUsageFlagBits::eUniformBuffer
				, vk::SharingMode::eExclusive
				, queueFamiliesIndex
			});
			const auto stagingBufferMemoryRequirement = device.getBufferMemoryRequirements(pixelToWorldSpaceTransformBuffer);
			pixelToWorldSpaceTransformBufferMemory = allocateMemory(device, physicalDevice, stagingBufferMemoryRequirement, true);
			device.bindBufferMemory(pixelToWorldSpaceTransformBuffer, pixelToWorldSpaceTransformBufferMemory, 0);
		};

		const auto initDescriptorPool = [&]()
		{
			// Describe the size of each used descriptor type and the number of descriptor sets (To check against the association/create layout step)
			const auto descriptorPoolSizes = std::vector<vk::DescriptorPoolSize>{
				{vk::DescriptorType::eCombinedImageSampler, 2} // Volume image and transfer image
				, {vk::DescriptorType::eStorageImage, 1} // Raycasted image
				, {vk::DescriptorType::eUniformBuffer, 2} // View pyramid struct and transform matrix for pixel to world space
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
				// View pyramid
				, vk::DescriptorSetLayoutBinding{
					3
					, vk::DescriptorType::eUniformBuffer
					, vk::ShaderStageFlagBits::eCompute
					, sampler
				}
				// Pixel to world space
				, vk::DescriptorSetLayoutBinding{
					4
					, vk::DescriptorType::eUniformBuffer
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
		setupViewPyramidStructBuffer();
		setupPixelToWorldSpaceTransformBuffer();

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

		// ViewPyramidBuffer
		auto descriptorBufferInfo = vk::DescriptorBufferInfo{
			viewPyramidStructBuffer
			, 0
			, sizeof(ViewPyramid)
		};
		device.updateDescriptorSets(vk::WriteDescriptorSet{
			descriptorSets.front()
			, 3
			, 0
			, vk::DescriptorType::eUniformBuffer
			, {}
			, descriptorBufferInfo
		}, {});

		// PixelToWorldSpaceBuffer
		descriptorBufferInfo = vk::DescriptorBufferInfo{
			pixelToWorldSpaceTransformBuffer
			, 0
			, sizeof(glm::mat4)
		};
		device.updateDescriptorSets(vk::WriteDescriptorSet{
			descriptorSets.front()
			, 4
			, 0
			, vk::DescriptorType::eUniformBuffer
			, {}
			, descriptorBufferInfo
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
	void preRenderLoop(const ApplicationInfo& applicationInfo)
	{
		APPLICATION_INFO_BINDINGS

		loadVolumeData();
		prepareHistogramVolumeData();
		validateShaders(shaderMap);
		setupViewPyramid();
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
	void updateShaderUniforms(const ApplicationInfo& applicationInfo)
	{
		APPLICATION_INFO_BINDINGS;

		// The height of the viewPyramid doesn't change
		auto viewPyramidBasis = glm::mat4{ // Make a copy of the viewPyramid, we will use the base bone made from the setupViewPyramid()
			viewPyramid.side
			, viewPyramid.up
			, viewPyramid.forward
			, glm::vec4{0, 0, 0, 1} // Location, is updated after this
		};

		// Construct a matrix encodes the rotated viewPyramid info and the rotated translation
		// These rotations will rotate the viewPyramid on top of its default transform via setupViewPyramid()
		//const auto verticalRotate = glm::rotate(glm::radians(-90.0f), glm::vec3{1, 0, 0}); // Pitch [-90, 90] 
		const auto verticalRotate = glm::rotate(verticalRadian, glm::vec3{1, 0, 0}); // Pitch [-90, 90] 
		const auto horizontalRotate = glm::rotate(horizontalRadian, glm::vec3{0, 0, 1}); // Roll [-180, 180]

		viewPyramidBasis = horizontalRotate * verticalRotate * viewPyramidBasis;
		const auto rotatedVolumeGridToCamera = horizontalRotate * verticalRotate * glm::vec4{static_cast<glm::vec3>(viewPyramid.location - volumeGridCenter), 1.0f};
		viewPyramidBasis[3] = glm::vec4{static_cast<glm::vec3>(volumeGridCenter + rotatedVolumeGridToCamera), 1.0f};

		const auto newViewPyramid = ViewPyramid{
			viewPyramid.height
			, viewPyramidBasis[3]
			, viewPyramidBasis[0]
			, viewPyramidBasis[1]
			, viewPyramidBasis[2]
		};
		void* memory = device.mapMemory(viewPyramidStructBufferMemory, 0, device.getBufferMemoryRequirements(viewPyramidStructBuffer).size);
		std::memcpy(memory, &newViewPyramid, sizeof(ViewPyramid));
		device.unmapMemory(viewPyramidStructBufferMemory);

		// Construct a pixelToWorldSpace transform using the matrix above
		const auto normalizeComponents = glm::mat4{ // [0, W) x [0, H) -> [0, 1]^2, subtract 1 because index goes from [0, extent) exclusive
			glm::vec4{1.0f / (surfaceExtent.width - 1.0f), 0, 0, 0}
			, glm::vec4{0, 1.0f / (surfaceExtent.height - 1.0f), 0, 0}
			, glm::vec4{0, 0, 1, 0}
			, glm::vec4{0, 0, 0, 1}
		};
		const auto viewportToWorld = glm::mat4{ // [0, 1]^2 -> x : [0, 1], y : [0, -1]
			glm::vec4{1, 0, 0, 0}
			, glm::vec4{0, -1, 0, 0}
			, glm::vec4{0, 0, 1, 0}
			, glm::vec4{0, 0, 0, 1}
		};
		const auto translatePreRotate = glm::translate(glm::vec3{-0.5, 0.5, 0}); // Shift x by -0.5, y by 0.5 (image grid to the center of the XY basis), and z by 1 (out of the screen). Center the image grid in the world basis
		const auto rotate = glm::mat4{ // Rotate to the direction based on the rotated view pyramid
			viewPyramidBasis[0]
			, viewPyramidBasis[1]
			, viewPyramidBasis[2]
			, glm::vec4{0, 0, 0, 1}
		};
		const auto translatePostRotate = glm::translate(glm::vec3(viewPyramidBasis[3]) + viewPyramid.height.x * glm::vec3{viewPyramidBasis[2]}); // Translate the pixel so that the image grid is centered with respect to the view pyramid
		// proxyImageOrigin   = (-0.5, 0.5, 0.0)
		// Image proxy width  = [-0.5, -0.5] = 1
		// Image proxy height =  [0.5, -0.5] = 1
		const auto pixelToWorldSpace = translatePostRotate * rotate * translatePreRotate * viewportToWorld * normalizeComponents;

		memory = device.mapMemory(pixelToWorldSpaceTransformBufferMemory, 0, device.getBufferMemoryRequirements(pixelToWorldSpaceTransformBuffer).size);
		std::memcpy(memory, &pixelToWorldSpace, sizeof(glm::mat4));
		device.unmapMemory(pixelToWorldSpaceTransformBufferMemory);
	}
	void renderCommands(const ApplicationInfo& applicationInfo, uint32_t imageIndex, bool isFirstFrame)
	{
		APPLICATION_INFO_BINDINGS

		updateTransferImage(applicationInfo);
		updateShaderUniforms(applicationInfo);

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

		// Don't copy the raycasted image to the swapchain since we don't want to render it directly to the viewport but through an imgui window
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
			, vk::ImageLayout::eShaderReadOnlyOptimal // Expected layout as descriptor for imgui fragment shader
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
			// TODO: if the extend is maxX, maxY. Must subtract it by 1 because index goes from [0, max) exclusive
			//TODO: ImGui::GetWindowSize() after this instead of getChildExtent()?
			const auto childWindowCursor = ImGui::GetCursorScreenPos(); // Must be placed above the visible button because we want the cursor of the current child window, not the button
			const auto drawList = ImGui::GetWindowDrawList();

			// Two default control points that aren't using the default control point color
			if (controlPoints.empty())
			{
				const auto leftMostControlPoint = unnormalizeCoordinate(ImVec2{0.0f, 0.0f}, ImVec2{childExtent.x - 1, childExtent.y - 1});
				controlPoints.push_back(ControlPoint{
					leftMostControlPoint
					, ImColor{0.0f, 0.0f, 0.0f, getAlpha(leftMostControlPoint, childExtent)
					}
				});
				const auto rightMostControlPoint = unnormalizeCoordinate(ImVec2{1.0f, 1.0f}, ImVec2{childExtent.x - 1, childExtent.y - 1});
				controlPoints.push_back(ControlPoint{
					rightMostControlPoint
					, ImColor{1.0f, 1.0f, 1.0f, getAlpha(rightMostControlPoint, childExtent)
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

				//if (i == 0) continue;
				//drawList->AddLine()
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
		// TODO: if the extend is maxX, maxY. Must subtract it by 1 because index goes from [0, max) exclusive
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
			if (ImGui::Button("Reset Control Points")) controlPoints.clear();
			ImGui::SameLine();
			if (ImGui::Button("Reset Rotations"))
			{
				horizontalRadian = 0.0f;
				verticalRadian = 0.0f;
			}
			ImGui::SliderAngle("Rotate Horizontal", &horizontalRadian, -180.0f, 180.0f);
			ImGui::SliderAngle("Rotate Vertical", &verticalRadian, -90.0f, 90.0f);

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
			ImGui::SetNextWindowSize(ImVec2{windowExtent.x - imguiWindowExtent.x, windowExtent.y}, ImGuiCond_Always);
			// https://github.com/ocornut/imgui/wiki/Image-Loading-and-Displaying-Examples#Example-for-Vulkan-users
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

		device.freeMemory(viewPyramidStructBufferMemory);
		device.destroyBuffer(viewPyramidStructBuffer);

		device.freeMemory(pixelToWorldSpaceTransformBufferMemory);
		device.destroyBuffer(pixelToWorldSpaceTransformBuffer);
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

