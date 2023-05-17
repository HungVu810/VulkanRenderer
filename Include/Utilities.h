#pragma once
#include "vulkan/vulkan.hpp"
#include "glm/vec2.hpp"
#include "glm/vec3.hpp"
#include "glm/vec4.hpp"
#include <filesystem>
#include <ranges>
#include <fstream>
#include <cassert>
#include <iostream>
#include <type_traits>
#include <functional>

namespace tag
{
	constexpr auto log = std::string_view{        "[[     LOG     ]] " };
	constexpr auto warning = std::string_view{    "[[---WARNING---]] " };
	constexpr auto exception = std::string_view{  "[[--EXCEPTION--]] " };
	constexpr auto error     = std::string_view{  "[[----ERROR----]] " };
	constexpr auto validation = std::string_view{ "[[  VALIDATES  ]] " };
}

namespace color
{
	constexpr auto white = vk::ClearColorValue{1.0f, 1.0f, 1.0f, 1.0f};
	constexpr auto black = vk::ClearColorValue{0.0f, 0.0f, 0.0f, 1.0f};
	constexpr auto red   = vk::ClearColorValue{1.0f, 0.0f, 0.0f, 1.0f};
	constexpr auto green = vk::ClearColorValue{0.0f, 1.0f, 0.0f, 1.0f};
	constexpr auto blue  = vk::ClearColorValue{0.0f, 0.0f, 1.0f, 1.0f};
}

namespace format
{
	struct Image{};
	struct Half{}; // Half-float, 16bit, available in glm but has a placeholder for now.
}

template<typename T>
constexpr auto toVulkanFormat()
{
	if (std::is_same_v<T, float>) return vk::Format::eR32Sfloat;
	else if (std::is_same_v<T, short>) return vk::Format::eR16Sint;
	else if (std::is_same_v<T, int>) return vk::Format::eR32Sint;
	else if (std::is_same_v<T, format::Half>) return vk::Format::eR16Sfloat;
	else if (std::is_same_v<T, format::Image>) {
		return vk::Format::eR8G8B8A8Unorm;
		// [0.0, 1.0]^4 == ([0, 255] / 255)^4. Using this format for the image because it's supported madatorily && contains the most format features
		// https://registry.khronos.org/vulkan/site/spec/latest/chapters/formats.html#VkFormat
	}
	else if (std::is_same_v<T, glm::vec2>) return vk::Format::eR32G32Sfloat;
	else if (std::is_same_v<T, glm::vec3>) return vk::Format::eR32G32B32Sfloat;
	else if (std::is_same_v<T, glm::vec4>) return vk::Format::eR32G32B32A32Sfloat;
	else return vk::Format::eUndefined;
}

// inlines/constexpr healpers
inline constexpr auto isAlpha = [](char c){return std::isalpha(c);};  // This is used as algorithm/ranges functor
inline constexpr auto print = [](const auto& in) { std::cout << in << '\n'; };
inline constexpr void assertm(bool condition, std::string_view message)
{
	if (!condition)
	{
		std::cerr << tag::error << message << '\n';
		assert(false);
	}
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
[[nodiscard]] inline void createFile(const std::filesystem::path& path)
{
	std::fstream{path, std::ios::out};
}
[[nodiscard]] inline auto getEnumeration(const auto& container)
{
	return std::views::zip(std::views::iota(0U, container.size()), container);
}
inline void submitCommandBufferOnceSynced(const vk::Device& device, const vk::Queue& queue, const vk::CommandBuffer& commandBuffer, const std::function<void(const vk::CommandBuffer& commandBuffer)>& commands) // Synced means the host will wait on the device queue to finish it works
{
	const auto waitFence = device.createFence(vk::FenceCreateInfo{});
	commandBuffer.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
	commands(commandBuffer);
	commandBuffer.end();
	queue.submit(vk::SubmitInfo{{}, {}, commandBuffer}, waitFence);
	std::ignore = device.waitForFences(waitFence, VK_TRUE, std::numeric_limits<uint64_t>::max());
	device.destroy(waitFence);
}
inline void checkVkResult(VkResult result) // For C-API
{
	if (result != VK_SUCCESS) throw std::runtime_error{"Failed to init ImGUI."};
};

// Normal helpers
[[nodiscard]] std::string getChecksum(const std::filesystem::path& path);


