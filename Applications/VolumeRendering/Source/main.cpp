#include "VulkanApplication.h"
#include "vulkan/vulkan.hpp" // Do not place this header above the VulkanApplication.h
#include "Shader.h"
#include "Allocation.h"
#include "imgui.h"
#include <thread> // In case of wanting more workers
#include <cstdint> // Needed for uint32_t
#include <limits> // Needed for std::numeric_limits
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

namespace 
{
	using Intensity = float; // Can't do short because sampler3D will always return a vec of floats

	// TODO: use this, imgui is disabled by default
	// ImGui
	constexpr auto USE_IMGUI = true;
	auto histogram = std::vector<float>(200); // For imgui representation of the intensities histogram. The size of the histogram is the number of samples.
	struct ControlPoint
	{
		ImVec2 position; // With respect to the histogram child window cursor
		ImColor color; // Picked from the HSV color wheel
	};
	auto controlPoints = std::vector<ControlPoint>{}; // Clicked control points, the position is 
	auto isImguiInit = false;

	// Application
	constexpr auto NUM_SLIDES = 113; // Ratio is 1:1:2, 2 unit of depth
	constexpr auto SLIDE_HEIGHT = 256;
	constexpr auto SLIDE_WIDTH = 256;
	constexpr auto NUM_INTENSITIES = NUM_SLIDES * SLIDE_HEIGHT * SLIDE_WIDTH;
	constexpr auto TOTAL_SCAN_BYTES = NUM_INTENSITIES * sizeof(Intensity);
	// z-y-x order, contains all intensity values of each slide images. Tightly packed
	// vector instead of array bececause NUM_INTENSITIES is large, occupy the heap instead
	auto intensities = Resource{std::vector<Intensity>(NUM_INTENSITIES), toVulkanFormat<Intensity>()}; // The actual data used for sampling
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

	// Private helpers
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

	// Private render functions
	void preRenderLoop(const ApplicationInfo& applicationInfo)
	{
		APPLICATION_INFO_BINDINGS

		loadVolumeData();
		prepareHistogramVolumeData();
		validateShaders(shaderMap);
		initComputePipeline(applicationInfo);
		submitCommandBufferOnceSynced(device, queue, commandBuffer, [&](const vk::CommandBuffer& commandBuffer){
			// Volume image layout: undefined -> transferDstOptimal, which is the expected layout when using the copy command
			commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, {vk::ImageMemoryBarrier{
				vk::AccessFlagBits::eNone
				, vk::AccessFlagBits::eTransferWrite
				, vk::ImageLayout::eUndefined
				, vk::ImageLayout::eTransferDstOptimal
				, VK_QUEUE_FAMILY_IGNORED // Same queue family, don't transfer the queue ownership
				, VK_QUEUE_FAMILY_IGNORED
				, volumeImage
				, vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
			}});
			// Copy the volume data from the staging buffer to the volume image used by the descriptor
			commandBuffer.copyBufferToImage(stagingBuffer, volumeImage, vk::ImageLayout::eTransferDstOptimal, vk::BufferImageCopy{
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
	// TODO: Create a struct RenderFrame in RunInfo that accept a recording function and an optional imgui commands function, will be check against the USE_IMGUI var
	// Remove isFirstFrame, this can be done in the preRenderLoop funcitno
	void recordRenderingCommands(const ApplicationInfo& applicationInfo, uint32_t imageIndex, bool isFirstFrame)
	{
		APPLICATION_INFO_BINDINGS

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
		commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, computePipelineLayout, 0, descriptorSets, {}); // 3D volume data
		const auto numInvocationPerX = 10; // Also put this number in the compute shader
		const auto numInvocationPerY = 10; // Also put this number in the compute shader
		assert((WIDTH % numInvocationPerX) == 0 && (HEIGHT % numInvocationPerY) == 0);
		commandBuffer.dispatch(WIDTH / numInvocationPerX, HEIGHT / numInvocationPerY, 1); // NOTE: group size must be at least 1 for all x, y, z

		const auto swapchainImage = device.getSwapchainImagesKHR(swapchain)[imageIndex];

		// Barrier to sync writting to raycastedImage via compute shader and copy it to swapchain image
		// Raycasted image layout: general -> transferSrc, before copy commmand
		// Swapchain image layout: undefined -> transferDst, before copy command
		commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, {vk::ImageMemoryBarrier{
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

		// Copy data from the rendered raycasted image via the compute shader to the current swapchain image
		commandBuffer.copyImage(raycastedImage, vk::ImageLayout::eTransferSrcOptimal, swapchainImage, vk::ImageLayout::eTransferDstOptimal, vk::ImageCopy{
				vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, 0, 0, 1}
				, {0, 0, 0}
				, vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, 0, 0, 1}
				, {0, 0, 0}
				, vk::Extent3D{surfaceExtent, 1}
		});

		// Transfer the swapchain image layout back to a presentable layout
		// Swapchain image layout: transferDst -> presentSrcKHR, before presented
		commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eBottomOfPipe, {}, {}, {}, vk::ImageMemoryBarrier{
				vk::AccessFlagBits::eTransferWrite // written during the copy command
				, vk::AccessFlagBits::eNone
				, vk::ImageLayout::eTransferDstOptimal
				, vk::ImageLayout::eColorAttachmentOptimal// For UI imgui renderpass. Will be transtioned into presentSrcKHR at the end of that renderpass
				, VK_QUEUE_FAMILY_IGNORED
				, VK_QUEUE_FAMILY_IGNORED
				, swapchainImage
				, vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
		});
	}
	// TODO: port utilities function from os project over
	void imguiCommands()
	{
		//ImGui::ShowDemoWindow();
		//ImGui::PlotHistogram("Intensity histogram", histogram.data(), histogram.size(), 0, nullptr, 0.0, 1.0, ImVec2{0, 80.0}); // Scale min/max represent the lowest/highest values of the data histogram
		//const auto& io = ImGui::GetIO();

		// Add 2 default control points at the min and max, the user can drag these 2 points but the x axis is fixed, only the y can be manipulated
		// Added control point will have the color black by default, click on
		// the cnotrol point ->  HSV color picker or enter color value rgb no A
		// -> interpolate the color of the control points then visualize it on
		// the color stripe on top of the canvas

		ImGui::Begin("Transfer function editor", 0, ImGuiWindowFlags_NoResize); // Create a window. No resizing because this is mess up the control point positions, temporary doing this for now.
		// Window default settings	
		if (!isImguiInit)
		{
			ImGui::SetWindowPos(ImVec2{0, 0});
			ImGui::SetWindowSize(ImVec2{500, 500});
		}

		// ------------- color wheel picker
		// Test with the 2 default control points with defined colors, then show the color  in the chilr window
		// TODO: Moving the control points around with left mouse when click on empty area, check if drawn circle can be dectedted with imgui function
		// Add/delete control points, when click on a contol point, a popup shown and prompt (change color or delete)
		// TODO: HSV color wheel when click with right mouse
		// TODO: seet widget color with the 2d color spectrum from the control points

		{
			// Data
			const auto drawList = ImGui::GetWindowDrawList();
			const auto colorSpectrumHeightPercentage = 1.0f / 12.0f; // With respect to the transfer function window size
			const auto colorSpectrumChildExtent = ImVec2{
				ImGui::GetContentRegionAvail().x
				, ImGui::GetContentRegionAvail().y * colorSpectrumHeightPercentage - ImGui::GetStyle().FramePadding.y / 2 // -1. Optionally subtract 1 to avoid scrolling appear when resizing the imgui window due to the widgets size perfectly matched the content region
			};
			const auto histogramAlphaHeightPercentage = 1.0f - colorSpectrumHeightPercentage;
			const auto histogramAlphaChildExtent = ImVec2{
				ImGui::GetContentRegionAvail().x
				, ImGui::GetContentRegionAvail().y * histogramAlphaHeightPercentage - ImGui::GetStyle().FramePadding.y / 2 // Frame padding divided by 2 because we have 2 widget
			};
			const auto defaultControlPointColor = ImColor{1.0f, 0.0f, 0.0f};
			const auto getAlpha = [](const ImVec2& controlPointPosition, const ImVec2& childWindowExtent) // The controlPointPosition is with respect to the child window cursor
			{
				return 1.0f - (controlPointPosition.y / childWindowExtent.y); // Control point's y = 0 (hit the child window's ceilling) means max opacity = 1
			};

			// Two default control points that aren't using the default control point color
			if (!isImguiInit)
			{
				const auto leftMostControlPoint = unnormalizeCoordinate(ImVec2{0.0f, 0.5f}, histogramAlphaChildExtent);
				controlPoints.push_back(ControlPoint{
					leftMostControlPoint
					, ImColor{
						1.0f
						, 0.0f
						, 0.0f
						, getAlpha(leftMostControlPoint, histogramAlphaChildExtent)
					}
				});
				const auto rightMostControlPoint = unnormalizeCoordinate(ImVec2{1.0f, 0.5f}, histogramAlphaChildExtent);
				controlPoints.push_back(ControlPoint{
					rightMostControlPoint
					, ImColor{
						0.0f
						, 1.0f
						, 0.0f
						, getAlpha(rightMostControlPoint, histogramAlphaChildExtent)
					}
				});
			}

			// Transfer function canvas for alpha control points and histogram
			{
				ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2{0, 0}); // Needed to avoid padding between the child window and the invisible button/the clip rect, tested with a visible button
				ImGui::PushStyleColor(ImGuiCol_ChildBg, IM_COL32(50, 50, 50, 255));
				ImGui::BeginChild("Histogram Alpha", histogramAlphaChildExtent, true, ImGuiWindowFlags_NoMove);
				const auto childWindowCursor = ImGui::GetCursorScreenPos(); // Must be placed above the visible button because we want the cursor of the current child window, not the button

				// Mouse position capture area, push back any captured position (control point) for drawinng
				ImGui::InvisibleButton("Input position capture", ImGui::GetContentRegionAvail(), ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight); 
				if (ImGui::IsItemClicked()) // Hovering the invisible button within the widget
				{
					const auto clickedPositionGLFW = ImGui::GetMousePos(); // With respect to the glfw Window
					const auto clickedPositionInChild = subtract(clickedPositionGLFW, childWindowCursor);
					controlPoints.push_back(ControlPoint{
						clickedPositionInChild
						, ImColor{
							defaultControlPointColor.Value.x
							, defaultControlPointColor.Value.y
							, defaultControlPointColor.Value.z
							, getAlpha(clickedPositionInChild, histogramAlphaChildExtent)
						}
					});
					std::ranges::sort(controlPoints
						, [](const ImVec2& a, const ImVec2& b){return a.x < b.x; }
						, [](const ControlPoint& controlPoint){return controlPoint.position;}
					); // Only sort the control points in an ascending order based on the x position
				}

				// Drawing area within the current child windoww
				drawList->PushClipRect(childWindowCursor, add(childWindowCursor, histogramAlphaChildExtent));

				// Draw histogram
				const auto stepSize = 1.0f / histogram.size();
				for (int i = 0; i < histogram.size(); i++)
				{
					const auto upperLeftExtentX = i * stepSize * histogramAlphaChildExtent.x;
					const auto upperLeftExtentY = histogramAlphaChildExtent.y * (1.0f - histogram[i]); // count == histogram[i] is normalized, we can use it to scale the extent.y. We do (1.0f - count) to force the scale 0.0 and 1.0 to be further/closert to the child top left origin

					const auto lowerRightExtentX = (i + 1) * stepSize * histogramAlphaChildExtent.x;
					const auto lowerRightExtentY = histogramAlphaChildExtent.y;

					const auto upperLeft = add(childWindowCursor, ImVec2{upperLeftExtentX, upperLeftExtentY});
					const auto lowerRight = add(childWindowCursor, ImVec2{lowerRightExtentX, lowerRightExtentY});
					drawList->AddRectFilled(upperLeft, lowerRight, ImColor{0.5f, 0.5f, 0.5f});
				}

				// Draw any control points and connect them via lines
				// Offset the control point to be with respect to the ImGui cursor instead of a child cursor
				auto offsetedControlPointPositions = controlPoints | std::views::transform([&](const ControlPoint& controlPoint){return add(childWindowCursor, controlPoint.position);}); // The alpha control points are with respect to the child cursor, we need to offset it with the cursor position
				for (int i = 0; i < offsetedControlPointPositions.size(); i++)
				{
					drawList->AddCircleFilled(offsetedControlPointPositions[i], 5, ImColor{1.0f, 1.0f, 1.0f}, 0);
					if (i == 0) continue;
					drawList->AddLine(offsetedControlPointPositions[i - 1], offsetedControlPointPositions[i], ImColor{1.0f, 1.0f, 1.0f});
				}
				drawList->PopClipRect();

				ImGui::EndChild();
				ImGui::PopStyleColor();
				ImGui::PopStyleVar();
			}

			// Transfer function color spectrum, another child window
			{
				ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2{0, 0}); // Needed to avoid padding between the child window and the clip rect, tested with a visible button
				ImGui::PushStyleColor(ImGuiCol_ChildBg, IM_COL32(50, 50, 50, 255));  // Set a background color
				ImGui::BeginChild("Color spectrum", colorSpectrumChildExtent, true, ImGuiWindowFlags_NoMove);
				const auto childWindowCursor = ImGui::GetCursorScreenPos();

				// Draw each of the color spectrum rectangles per 2 control points
				drawList->PushClipRect(childWindowCursor, add(childWindowCursor, colorSpectrumChildExtent));

				for (size_t i = 1; i < controlPoints.size(); i++)
				{
					const auto upperLeftPosition = ImVec2{childWindowCursor.x + controlPoints[i - 1].position.x, childWindowCursor.y};
					const auto leftColor = controlPoints[i - 1].color; // Including alpha

					const auto lowerRightPosition = ImVec2{childWindowCursor.x + controlPoints[i].position.x, childWindowCursor.y + colorSpectrumChildExtent.y}; // Adding y of the child extent to reach to the floor of the current child window
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

				ImGui::EndChild();
				ImGui::PopStyleColor();
				ImGui::PopStyleVar();
			}
		}

		ImGui::End();

		isImguiInit = true;
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
		, preRenderLoop
		, recordRenderingCommands
		, imguiCommands
		, postRenderLoop
		, "Volume Rendering"
	};
	application.run(runInfo);

	return EXIT_SUCCESS;
}

