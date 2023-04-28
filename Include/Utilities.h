#pragma once
#include <filesystem>
#include <fstream>
#include <cassert>
#include <iostream>
#include <ranges>

namespace tag
{
	constexpr auto log = std::string_view{        "[[     LOG     ]] " };
	constexpr auto warning = std::string_view{    "[[---WARNING---]] " };
	constexpr auto exception = std::string_view{  "[[--EXCEPTION--]] " };
	constexpr auto error     = std::string_view{  "[[----ERROR----]] " };
	constexpr auto validation = std::string_view{ "[[  VALIDATES  ]] " };
}

// inlines/constexpr healpers
constexpr auto isAlpha = [](char c){return std::isalpha(c);};  // This is used as algorithm/ranges functor
constexpr auto print = [](const auto& in) { std::cout << in << '\n'; };
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

// Normal helpers
[[nodiscard]] std::string getChecksum(const std::filesystem::path& path);


