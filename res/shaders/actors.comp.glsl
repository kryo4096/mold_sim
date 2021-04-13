#version 450

#define M_PI 3.1415926535897932384626433832795


layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

struct Actor {
    vec2 position;
    float angle;
};

layout(set = 0, binding = 0, rg32f) uniform image2D back_buf;
layout(set = 0, binding = 1, std430) buffer buf {
    Actor[] actor_data;
};

layout(set=0, binding = 2) uniform Data {
    uint actor_count;
    float delta_time;
    float time;
    bool init;
    float sensor_angle;
    float sensor_distance;     
    int sensor_size;
    float actor_speed;
    float phero_strength;
    float turn_speed;
    float turn_gamma;
    float randomness;
    float init_radius;
    float relative_angle;
    float random_angle;
    float init_gamma;
} u;


float rand(vec2 n) { 
	return fract(sin(dot(n, vec2(12.9898, 4.1414))) * 43758.5453);
}

float random(float seed) {
    return rand(vec2(gl_GlobalInvocationID.x / 1000., u.time + seed));
}

float sense(vec2 position, int size) {
    float r = 0;

    for(int x = -size / 2; x <= size / 2; x++) {
        for(int y = -size / 2; y <= size / 2; y++) {
            r += imageLoad(back_buf, ivec2(position + vec2(x,y))).x;
        }
    }

    return r;
}

vec2 astep(vec2 position, float angle, float step_size) {
    return position + vec2(cos(angle), sin(angle)) * step_size;
}

void init_actors() {
    ivec2 bounds = imageSize(back_buf);
    Actor actor;

    float angle = float(gl_GlobalInvocationID.x) / float(u.actor_count) * 64 * M_PI;

    float rand =  random(1337);
    
    actor.position = min(max(vec2(cos(angle), sin(angle)) * bounds.y * pow(0.0001 + rand * (1-0.0001), u.init_gamma) * u.init_radius * 0.5 + bounds / 2, vec2(0)), vec2(bounds));

    actor.angle = angle + u.relative_angle + u.random_angle * (random(2342) - 0.5);
    
    actor_data[gl_GlobalInvocationID.x] = actor;
}

void main() {

    if (u.init) {
        init_actors();
        return;
    }
  
    if(gl_GlobalInvocationID.x > u.actor_count) return;

   
   
    Actor a = actor_data[gl_GlobalInvocationID.x];

    float ls = sense(astep(a.position, a.angle + u.sensor_angle, u.sensor_distance), u.sensor_size);
    float fs = sense(astep(a.position, a.angle, u.sensor_distance), u.sensor_size);
    float rs = sense(astep(a.position, a.angle - u.sensor_angle, u.sensor_distance), u.sensor_size);

    float delta_angle = 0;

    if(fs > ls && fs > rs) {
        delta_angle = 0;
    } else if (fs < ls && fs < rs) {
        delta_angle = (random(13132) - 0.5) * 4 * (rs + ls) * 0.02;
    } else if (rs > ls) {
        delta_angle = -pow(rs, u.turn_gamma);
    } else if (ls > rs) {
        delta_angle = pow(ls, u.turn_gamma);
    }

    delta_angle += (random(131232) - 0.5) * u.randomness * 2;

    a.angle += delta_angle * u.delta_time * u.turn_speed;

    vec2 p = astep(a.position, a.angle, u.delta_time * u.actor_speed);

    ivec2 bounds = imageSize(back_buf);

    if(p.x > 1 && p.y > 1 && p.x < bounds.x-2 && p.y < bounds.y-2) {
        a.position = p;

        vec4 f = imageLoad(back_buf, ivec2(a.position));

        f.x += u.delta_time * u.phero_strength;

        barrier();

        imageStore(back_buf, ivec2(a.position), f);
    
    } else {
        a.angle = random(134234) * 2 * M_PI;
    }

    actor_data[gl_GlobalInvocationID.x] = a;
    
}