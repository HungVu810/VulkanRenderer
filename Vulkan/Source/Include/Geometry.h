#pragma once
#include <vulkan/vulkan.hpp>
#include <glm/glm.hpp>
#include <vector>
#include <array>

struct Vertex
{
	glm::vec3 position;
	glm::vec3 normal;
	glm::vec3 color; // RGB
	// std::variant<glm::vec3, glm::vec2> color/textcoord ?
	// glm::vec2 textcoord;

	consteval static auto getVertexInputBindingDescription()
	{
		return vk::VertexInputBindingDescription{
			0,
			sizeof(Vertex),
		};
	}

	consteval static auto getVertexInputAttributeDescription()
	{
		 return std::array<vk::VertexInputAttributeDescription, 3>{
			 vk::VertexInputAttributeDescription{0, 0, vk::Format::eR32G32B32A32Sfloat, offsetof(Vertex, position)}
			 , {1, 0, vk::Format::eR32G32B32A32Sfloat, offsetof(Vertex, normal)}
			 , {2, 0, vk::Format::eR32G32B32A32Sfloat, offsetof(Vertex, color)}
		 };
	}
};

// each model's mesh are drawn independantly since drawing the whole model messed up the texture
struct Mesh{
	// store continous vertices data. position (3), normal (3), texcoord(2) per vertex
	std::vector<Vertex> vertices;
	//// ebo buffer, stores continous faces indicies. 3 indices per face
	//std::vector<unsigned> ebuf;
	//// textures for this mesh
	//std::vector<std::shared_ptr<gl_texture>> material;
	//// shininess level for the mesh's phong-specularity calculation
	//float shininess = -1.0f;
};
