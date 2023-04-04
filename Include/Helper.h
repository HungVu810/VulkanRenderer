#pragma once
#include <filesystem>
#include <fstream>
#include "Prompt.h"

// inlines/constexpr healpers
constexpr auto isAlpha = [](char c){return std::isalpha(c);};  // This is used as algorithm/ranges functor
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

// Normal helpers
[[nodiscard]] std::string getChecksum(const std::filesystem::path& path);


