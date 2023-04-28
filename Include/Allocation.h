#pragma once
#include <vulkan/vulkan.hpp>
#include <algorithm>
#include <tuple>
#include "Utilities.h"

// https://registry.khronos.org/vulkan/site/spec/latest/chapters/resources.html
// 1. device.CreateResource -> logical storage
// 2. device.GetResourceMemoryRequirement
// 3. allocateMemory() -> actual physical memory
// 4. device.BindResourceMemory
// Buffer
	// 5. device.mapMemory()
	// 6. std::memcpy
	// 7. device.unmapMemory(0
	// Bind the logical buffer to the allocated memory and upload the buffer data
	// to the allocated memory
// Image
	// 5. vkCreateImageView

// Physical memory allocation for the logical buffer
[[nodiscard]] auto allocateMemory(const vk::Device& device, const vk::PhysicalDevice& physicalDevice, const vk::MemoryRequirements& memoryRequirements)
{
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
	return device.allocateMemory(memoryAllocateInfo);
}



