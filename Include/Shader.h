#pragma once
#include <vulkan/vulkan.hpp>
#include <vector>
#include <string>
#include <filesystem>
#include <unordered_map>

class Shader
{
	public:
		~Shader();

		[[nodiscard]] std::vector<uint32_t> getBinaryData() const;

		[[nodiscard]] vk::ShaderStageFlagBits getStage() const noexcept;

		friend void validateShaders(std::unordered_map<std::string, Shader>& shaderMap);

	private:
		Shader(std::string_view shaderFileName);

		std::filesystem::path path;
};

void validateShaders(std::unordered_map<std::string, Shader>& shaderMap); // Instantiate Shader objects during validation process

namespace
{
	[[nodiscard]] inline auto getShaderBinaryData(std::unordered_map<std::string, Shader>& shaderMap, const std::string& shaderName)
	{
		//return shaderMap.find(shaderName)->second.getBinaryData();
		const auto binaryData = shaderMap.find(shaderName)->second.getBinaryData();
		// return std::span<uint32_t>{(uint32_t*)binaryData.data(), binaryData.size()};
		return binaryData;
	}
};


