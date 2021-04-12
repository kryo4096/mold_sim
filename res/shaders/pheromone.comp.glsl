#version 450

#define M_PI 3.1415926535897932384626433832795

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(push_constant) uniform PushConstantData {
    float delta_time;
    bool init_image;
    float diffusion_constant;
    float dissipation_constant;
    float time;
} u;


layout(set = 0, binding = 0, rg32f) uniform image2D back_buf;
layout(set = 0, binding = 1, rg32f) uniform image2D render_buf;

const mat3 laplacian = mat3(1,1,1,1,-8,1,1,1,1);

void main() {
    ivec2 bounds = imageSize(back_buf);
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);


    if(u.init_image) {
   
        imageStore(back_buf, pos, vec4(0));
        imageStore(render_buf, pos, vec4(0));

        return;
    }

    if(pos.x > 0 && pos.y > 0 && pos.x < bounds.x-1 && pos.y < bounds.y-1) {

        float l = 0;

        for(int i = -1; i <= 1; i++) {
            for(int j = -1; j <= 1; j++) {
                l += imageLoad(back_buf, ivec2(pos.x + i, pos.y + j)).x * laplacian[i+1][j+1]; 
            }
        }

        vec4 pixel = imageLoad(back_buf, pos);

        pixel.x += (u.diffusion_constant*l - u.dissipation_constant * pixel.x) * u.delta_time;

        imageStore(render_buf, pos, pixel);

    }
}