#version 450

layout(location = 0) in vec2 position;
layout(location = 0) out vec2 tex_coords;

layout(set = 1, binding = 0) uniform Data {
    vec2 zoom_pos;
    float zoom;
} u;

void main() {
    
    
    gl_Position = vec4(position, 0.0, 1.0);


    tex_coords = u.zoom_pos + (position) / 2 / u.zoom;
}