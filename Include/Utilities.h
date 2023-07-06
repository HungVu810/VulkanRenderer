#pragma once
#include "vulkan/vulkan.hpp"
#include "glm/vec2.hpp"
#include "glm/vec3.hpp"
#include "glm/vec4.hpp"
#include "imgui.h"
#include <filesystem>
#include <ranges>
#include <algorithm>
#include <numeric>
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

namespace math
{
	// https://glm.g-truc.net/0.9.2/api/index.html
	const auto PI = std::acos(-1.0f);
	inline auto toRadian(float angle) {return angle * PI / 180.0;}
	inline auto toAngle(float radian) {return radian * 180 / PI;}
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

/* Inline/Lambda helpers ======================================== */
// Lambdas
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

// General
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
template<typename AccumulatedType> // The type of the values produced from the accumulated chunks
[[nodiscard]] inline auto accumulateChunks(const auto& container, int width = 1) // {0, 1, 2, 3, 4, 5} -> {6, 9} with width = 4, ie 2 accumulated chunks {0, 1, 2, 3} and {4, 5}
{
	assert(width > 0 && width <= container.size());
	return container
		| std::views::chunk(width)
		| std::views::transform([](const auto& chunk){return std::accumulate(chunk.begin(), chunk.end(), static_cast<AccumulatedType>(0)); }) // Cast 0 to double instead of the 0 (int) to prevent the accumulate from casting the value to int
		| std::ranges::to<std::vector>();
}
template<typename ComparedType>
[[nodiscard]] inline auto sortAndRemoveDuplicates(const auto& container)
{
	// https://en.cppreference.com/w/cpp/algorithm/ranges/unique
	// Sorting first (ascending order) will allow 100% removal of the duplicates
	auto targetContainer = container;
	std::ranges::sort(targetContainer, [](ComparedType a, ComparedType b){return a < b;});
	const auto erasableRange = std::ranges::unique(targetContainer);
	targetContainer.erase(erasableRange.begin(), erasableRange.end()); // Will remove the excessive indeterminates at the end
	return targetContainer;
}

// TODO: make these funciton toGlmVec#(ImVec#), and toImVec#(GlmVec#) so we can use glm operations +,-... without the below?
// ImGui
[[nodiscard]] inline auto toVec2(const ImVec2& a)
{
	return glm::vec2{a.x, a.y};
}
[[nodiscard]] inline auto toVec3(const ImVec2& a, float z)
{
	return glm::vec3{a.x, a.y, z};
}
[[nodiscard]] inline auto toVec3(const ImVec4& a)
{
	return glm::vec3{a.x, a.y, a.z};
}
[[nodiscard]] inline auto toVec3(const ImColor& a)
{
	return glm::vec3{a.Value.x, a.Value.y, a.Value.z};
}
[[nodiscard]] inline auto toVec4(const ImVec4& a)
{
	return glm::vec4{a.x, a.y, a.z, a.w};
}
[[nodiscard]] inline auto toVec4(const ImColor& a)
{
	return glm::vec4{a.Value.x, a.Value.y, a.Value.z, a.Value.w};
}

[[nodiscard]] inline auto add(const ImVec2& a, const ImVec2& b)
{
	return ImVec2{a.x + b.x, a.y + b.y};
}
[[nodiscard]] inline auto add(const ImVec4& a, const ImVec4& b)
{
	return ImVec4{a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}
[[nodiscard]] inline auto subtract(const ImVec2& a, const ImVec2& b)
{
	return ImVec2{a.x - b.x, a.y - b.y};
}
[[nodiscard]] inline auto subtract(const ImVec4& a, const ImVec4& b)
{
	return ImVec4{a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w};
}
[[nodiscard]] inline auto scale(const ImVec2& a, const float k)
{
	return ImVec2{a.x * k, a.y * k};
}
[[nodiscard]] inline auto scale(const ImVec4& a, const float k)
{
	return ImVec4{a.x * k, a.y * k, a.z * k, a.w * k};
}
[[nodiscard]] inline auto length(const ImVec2& a)
{
	return std::sqrt(a.x * a.x + a.y * a.y);
}
[[nodiscard]] inline auto unnormalizeCoordinate(const ImVec2& coordinate, const ImVec2& extent)
{
	assertm(0 <= coordinate.x && coordinate.x <= 1
		 && 0 <= coordinate.y && coordinate.y <= 1, "Invalid coordinate");
		 //&& 0 < extent.x && 0 < extent.y, "Invalid coordinate/extent");
	return ImVec2{coordinate.x * extent.x, coordinate.y * extent.y};
};
/* Inline/Lambda helpers ======================================== */

/* Normal helpers =============================================== */
// Shader
[[nodiscard]] std::string getChecksum(const std::filesystem::path& path);
/* Normal helpers =============================================== */

