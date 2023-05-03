#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 color;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec3 fragNormal;

void main() {
	gl_Position = vec4(position, 1.0);
	fragColor = color;
	fragNormal = normal;
}
