#include <filesystem>
#include <fstream>
#include <unordered_map>
#include <string>
#include <vulkan/vulkan.hpp>
#include "Prompt.h"
#include "Helper.h"
#include "Shader.h"

namespace
{
	[[nodiscard]] inline auto getShaderCompilerPath()
	{
		return std::filesystem::absolute(SHADER_COMPILER);
	}
	[[nodiscard]] inline auto getShaderValidatorPath()
	{
		return std::filesystem::path{SHADER_VALIDATOR};
	}
	[[nodiscard]] inline auto getShaderPath(std::string_view shaderFileName)
	{
		return std::filesystem::absolute(std::filesystem::path(SHADER_SOURCE)/shaderFileName);
	}
	[[nodiscard]] auto getShaderStage(const std::filesystem::path& shaderPath) // Used by Shader class after "path" is initialized
	{
		const auto extension = shaderPath.extension();
		auto returnValue = std::optional<vk::ShaderStageFlagBits>{std::nullopt};
		if (extension == ".vert") returnValue = vk::ShaderStageFlagBits::eVertex;
		else if (extension == ".tesc") returnValue = vk::ShaderStageFlagBits::eTessellationControl;
		else if (extension == ".tese") returnValue = vk::ShaderStageFlagBits::eTessellationEvaluation;
		else if (extension == ".geom") returnValue = vk::ShaderStageFlagBits::eGeometry;
		else if (extension == ".frag") returnValue = vk::ShaderStageFlagBits::eFragment;
		else if (extension == ".comp") returnValue = vk::ShaderStageFlagBits::eCompute;
		return returnValue;
	}
	[[nodiscard]] inline auto getShaderBinaryPath(const std::filesystem::path& shaderPath) // Used by Shader class after "path" is initialized
	{
		return std::filesystem::absolute(std::filesystem::path(SHADER_BINARY)/shaderPath.filename().concat(".spv"));
	}
	[[nodiscard]] auto getShaderCompileCommand(const std::filesystem::path& shaderPath) // Used by validateShaders
	{
		const auto shaderPathString = unquotePathString(shaderPath);
		const auto compilerPathString = unquotePathString(getShaderCompilerPath());
		const auto binaryPathString = unquotePathString(getShaderBinaryPath(shaderPath));
		auto command = std::stringstream{};
		command << compilerPathString << " " << shaderPathString << " -o " << binaryPathString;
		return command.str();
	}
	[[nodiscard]] inline auto compileShader(const std::filesystem::path& shaderPath)
	{
		const auto compileCommand = getShaderCompileCommand(shaderPath);
		return std::system(compileCommand.data());
	}
}

void validateShaders(std::unordered_map<std::string, Shader>& shaderMap)
{
	shaderMap.clear();
	const auto validationPath = getShaderValidatorPath();
	auto shouldValidateShaders = false;
	auto lastChecksums = std::vector<std::string>{};
	if (!std::filesystem::exists(validationPath))
	{
		createFile(validationPath);
	}
	auto validationFile = std::fstream{validationPath, std::ios::in}; // Read the validation contents then clear.
	if (!std::filesystem::is_empty(validationPath))
	{
		auto checkSum = std::string{};
		while (std::getline(validationFile, checkSum))
		{
			lastChecksums.push_back(checkSum);
		}
		std::fstream{validationPath, std::ios::trunc}; // Clear file contents
		shouldValidateShaders = true;
	}
	validationFile = std::fstream{validationPath, std::ios::out}; // Only write to validation after this.
	const auto compileAndWriteChecksum = [&](const std::filesystem::path& shaderPath, std::string_view checksum)
	{
		std::cout << tag::warning << "Compiling shader at " << shaderPath << '\n';
		if (compileShader(shaderPath) == 0) // Success
		{
			validationFile << checksum << std::endl;
			std::cout << tag::warning << "Done\n";
		}
		else std::cout << tag::error << ">>> Failed\n";
	};
	for (const auto& file : std::filesystem::directory_iterator{SHADER_SOURCE})
	{
		if (file == validationPath) continue;
		const auto checksum = getChecksum(file);
		if (!shouldValidateShaders) // Empty checksum file
		{
			compileAndWriteChecksum(file, checksum);
		}
		else
		{
			const auto iterChecksum = std::ranges::find(lastChecksums, checksum);
			if (iterChecksum != lastChecksums.end())
			{
				std::cout << tag::log << "Shader at " << file << " has no modifications\n";
				validationFile << checksum << std::endl;
			}
			else // Either the checksum can't be found due to file modification or this is a new shader
			{
				compileAndWriteChecksum(file, checksum);
			}
		}
		const auto shaderFileName = file.path().filename().string();
		// NOTE: Inserted for both compilable and uncompilable shaders
		shaderMap.insert({shaderFileName, Shader{shaderFileName}});
	}
}

Shader::Shader(std::string_view shaderFileName) : path{getShaderPath(shaderFileName)}{};

Shader::~Shader(){};

[[nodiscard]] std::vector<char> Shader::getBinaryData() const
{
	const auto binaryPath = getShaderBinaryPath(path);
	if (!std::filesystem::exists(binaryPath)) throw std::runtime_error{std::string{"Attempting to reference a non-existed shader binary at "} + path.string()};
	auto binaryFile = std::ifstream{binaryPath, std::ios::binary};
	auto bindaryData = std::vector<char>(std::filesystem::file_size(binaryPath));
	binaryFile.read(bindaryData.data(), bindaryData.size());
	return bindaryData;
}

[[nodiscard]] vk::ShaderStageFlagBits Shader::getStage() const noexcept
{
	return getShaderStage(path).value();
}

