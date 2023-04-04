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

		[[nodiscard]] std::vector<char> getBinaryData() const;

		[[nodiscard]] vk::ShaderStageFlagBits getStage() const noexcept;

		friend void validateShaders(std::unordered_map<std::string, Shader>& shaderMap); // Instantiate Shader objects during validation process

	private:
		Shader(std::string_view shaderFileName);

		std::filesystem::path path;
};

void validateShaders(std::unordered_map<std::string, Shader>& shaderMap);


