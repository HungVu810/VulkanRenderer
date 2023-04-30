#pragma once
#include <vulkan/vulkan.hpp>
#include <glm/glm.hpp>
#include <vector>
#include <optional>
#include <array>
#include "Utilities.h"

struct Vertex
{
	glm::vec3 position;
	glm::vec3 normal;
	glm::vec3 color;
	// std::variant<glm::vec3, glm::vec2> color/textcoord ?
	// glm::vec2 textcoord;
};

class VertexBuffer{
public:
	VertexBuffer(std::vector<Vertex> inVertices, std::optional<std::vector<uint32_t>> inElementBuffer = std::nullopt)
		: bindingNumber{-1}
		, vertices{inVertices}
		, elementBuffer{inElementBuffer}
		, buffer{}
		, memory{}
	{}

	~VertexBuffer()
	{

	}

private:
	int bindingNumber; // The binding number for this vertex buffer
	std::vector<Vertex> vertices; // Stored on the CPU. The data should be a continuous stream of attributes for cache-locality.
	std::optional<std::vector<uint32_t>> elementBuffer; // Stores faces' indicies. 3 indices per face

	vk::Device device; // Logical device from Vulkan Application
	vk::Buffer buffer; // Logical buffer on the logical device
	vk::DeviceMemory memory; // Allocated memory from the physical device by for the buffer
};

// Each model's mesh are drawn independantly since drawing the whole model messed up the texture
//class Mesh
//{
//	// vertex buffer
//
//	//// textures for this mesh
//	//std::vector<std::shared_ptr<gl_texture>> material;
//
//	//// shininess level for the mesh's phong-specularity calculation
//	//float shininess = -1.0f;
//};

class GeometryFactory
{
	// line
	// square, triangle, circle
	// cube, prism, sphere
	// common scenes (castle, standford room)
	// common props (bunny, teapot,..)
};


