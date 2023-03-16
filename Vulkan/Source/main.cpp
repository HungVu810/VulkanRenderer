#include <VulkanApplication.h>
#include <Geometry.h>
// imgui, imguizmo
// constexpr, consteval

int main() {
	// clip space
	//const auto triangle = std::vector<Vertex>{
	//	{{0.5f, 0.5f, 0.0f}, {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}}
	//	, {{-0.5f, 0.5f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}}
	//	, {{0.0f, -0.5f, 0.0f}, {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}}
	//};

	//const auto vertexInputBindingDescription = Vertex::getVertexInputBindingDescription();
	//const auto vertexInputAttributeDescription = Vertex::getVertexInputAttributeDescription();
	//vk::PipelineVertexInputStateCreateInfo VertexInputStateCreateInfo{
	//	{}
	//	, vertexInputBindingDescription
	//	, vertexInputAttributeDescription
	//};
	//vk::PipelineInputAssemblyStateCreateInfo InputAssemblyStateCreateInfo{

	//};

	//vk::BufferCreateInfo vertexBuffer{
	//	{}
	//	, triangle.size()
	//	, vk::BufferUsageFlagBits::eVertexBuffer
	//	, vk::SharingMode::eExclusive
	//	,
	//};

	VulkanApplication app;
	app.run();
	return EXIT_SUCCESS;
}


