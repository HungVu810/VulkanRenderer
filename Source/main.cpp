#include <VulkanApplication.h>
// imgui, imguizmo
// constexpr, consteval

// create a appThread class that takes works and assigned with enum of the current work

// TODO: seperate this vuklan application into a framework to support different
// type of graphic program, ie volumn rendering, normal mesh renderng

int main()
{
	VulkanApplication app;
	app.run();
	return EXIT_SUCCESS;
}


