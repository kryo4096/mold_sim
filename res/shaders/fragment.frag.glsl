#version 450


layout(location = 0) in vec2 tex_coords;
layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 0) uniform sampler2D tex;

layout(set = 0, binding = 1) uniform Data {
    float hue;
    float gamma;
    float brightness;
} u;

vec3 hsv2rgb(vec3 c)
{
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {
    vec4 pixel = texture(tex, tex_coords);
    
    f_color = vec4(0);

    float scale = 20. / u.brightness;

    //f_color = vec4(hsv2rgb(vec3(0.5 + x / scale, 1.0, 0.3 * abs(x / scale * 2.))), 1);

    float x = pow(pixel.x, u.gamma);

    //f_color = vec4(x / scale);

    f_color = vec4(x/scale * hsv2rgb(vec3(u.hue + 0.1 * x / scale, 1.0, 1.0)), 1.0);
    if(abs(x) > scale) {
        f_color.rgb += vec3(tanh((abs(x) - scale)/scale));
    }
    

    

}