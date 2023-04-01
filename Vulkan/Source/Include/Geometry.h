#pragma once
#include <vulkan/vulkan.hpp>
#include <glm/glm.hpp>
#include <vector>
#include <array>

namespace format
{
	constexpr auto floatType = vk::Format::eR32Sfloat;
	constexpr auto vec2 = vk::Format::eR32G32Sfloat;
	constexpr auto vec3 = vk::Format::eR32G32B32Sfloat;
	constexpr auto vec4 = vk::Format::eR32G32B32A32Sfloat;
}

struct Vertex
{
	glm::vec3 position;
	glm::vec3 normal;
	glm::vec3 color;
	// std::variant<glm::vec3, glm::vec2> color/textcoord ?
	// glm::vec2 textcoord;
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

class GeometryFactory
{
	// line
	// square, triangle, circle
	// cube, prism, sphere
	// common scenes (castle, standford room)
	// common props (bunny, teapot,..)
};
