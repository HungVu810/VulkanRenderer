#pragma once
#include <vulkan/vulkan.hpp>
#include <algorithm>
#include <tuple>
#include "Utilities.h"

// https://registry.khronos.org/vulkan/site/spec/latest/chapters/resources.html
// 1. device.CreateResource -> logical storage
// 2. device.GetResourceMemoryRequirement
// Buffer
	// 3. allocateMemory(TRUE) -> actual physical memory, need host access to the memory (mapMemory) to do memcpy
	// 4. void* = device.mapMemory() -> return a temporary host memory that mapped to the device's memory
	// 5. std::memcpy resource to void* of the host
	// 6. device.unmapMemory() -> transfer the host memory to device's and unmap
	// Bind the logical buffer to the allocated memory and upload the buffer data
	// to the allocated memory
// Image
	// 3. allocateMemory(FALSE) -> actual physical memory
	// 4. vkCreateImageView
// * device.BindResourceMemory

// Physical memory allocation for the logical buffer
[[nodiscard]] auto allocateMemory(const vk::Device& device, const vk::PhysicalDevice& physicalDevice, const vk::MemoryRequirements& memoryRequirements, bool shouldHostAccess = true)
{
	const auto memoryProperties = physicalDevice.getMemoryProperties();
	const auto indexedMemoryTypes = getEnumeration(memoryProperties.memoryTypes);
	const auto isSuitable = [&](const auto& indexedMemoryType)
	{
		const auto& [memoryIndex, memoryType] = indexedMemoryType;
		const auto memoryTypeBits = (1 << memoryIndex); // The type represented as bits, each type is counted as a power of 2 from 0
		const auto hasRequiredMemoryType = memoryRequirements.memoryTypeBits & memoryTypeBits;
		const auto hasRequiredMemoryProperty = memoryType.propertyFlags & (shouldHostAccess ? vk::MemoryPropertyFlagBits::eHostVisible : vk::MemoryPropertyFlagBits::eDeviceLocal);
		return hasRequiredMemoryType && hasRequiredMemoryProperty;
	};
	const auto iterIndexedMemoryTypes = std::ranges::find_if(indexedMemoryTypes, isSuitable);
	if (iterIndexedMemoryTypes == indexedMemoryTypes.end()) throw std::runtime_error{"Failed to find suitable memory type"};
	const auto memoryAllocateInfo = vk::MemoryAllocateInfo{
		memoryRequirements.size
		, std::get<0>(*iterIndexedMemoryTypes)
	};
	return device.allocateMemory(memoryAllocateInfo);
}



