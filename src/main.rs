use std::sync::Arc;
use std::vec::Vec;
use std::{
    error::Error,
    time::{Duration, Instant},
};
use std::{f32::consts, mem};

use rand::prelude::*;
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, CpuBufferPool, DeviceLocalBuffer},
    command_buffer::{AutoCommandBufferBuilder, CommandBuffer, DynamicState, SubpassContents},
    descriptor::{descriptor_set::PersistentDescriptorSet, PipelineLayoutAbstract},
    device::{Device, DeviceExtensions, Features},
    format::{ClearValue, Format},
    framebuffer::{Framebuffer, FramebufferAbstract, RenderPassAbstract, Subpass},
    image::{Dimensions, ImageUsage, ImageViewAccess, StorageImage, SwapchainImage},
    instance::{Instance, PhysicalDevice},
    pipeline::{viewport::Viewport, ComputePipeline, GraphicsPipeline},
    sampler::{Filter, MipmapMode, Sampler, SamplerAddressMode},
    swapchain::{
        self, AcquireError, Capabilities, ColorSpace, FullscreenExclusive, PresentMode, Surface,
        SurfaceTransform, Swapchain, SwapchainCreationError,
    },
    sync::{self, FlushError, GpuFuture},
};
use vulkano_win::VkSurfaceBuild;

use winit::{
    event::{
        ElementState, Event, KeyboardInput, MouseButton, MouseScrollDelta, VirtualKeyCode,
        WindowEvent,
    },
    event_loop::{ControlFlow, EventLoop},
    window::{Fullscreen, Window, WindowBuilder},
};

use egui::{FontDefinitions, Slider};
use egui_winit_platform::*;
use rustop::opts;

mod phero_cs {
    const _RECOMPILE_DUMMY: &str = include_str!("../res/shaders/pheromone.comp.glsl");

    vulkano_shaders::shader! {
        ty: "compute",
        path: "res/shaders/pheromone.comp.glsl"
    }
}

mod actors_cs {
    const _RECOMPILE_DUMMY: &str = include_str!("../res/shaders/actors.comp.glsl");

    vulkano_shaders::shader! {
        ty: "compute",
        path: "res/shaders/actors.comp.glsl"
    }
}

mod vs {
    const _RECOMPILE_DUMMY: &str = include_str!("../res/shaders/vertex.vert.glsl");

    vulkano_shaders::shader! {
        ty: "vertex",
        path: "res/shaders/vertex.vert.glsl"
    }
}

mod fs {
    const _RECOMPILE_DUMMY: &str = include_str!("../res/shaders/fragment.frag.glsl");

    vulkano_shaders::shader! {
        ty: "fragment",
        path: "res/shaders/fragment.frag.glsl"
    }
}

#[derive(Default, Copy, Clone)]
struct Vertex {
    position: [f32; 2],
}

impl Vertex {
    const fn new(x: f32, y: f32) -> Self {
        Self { position: [x, y] }
    }
}

vulkano::impl_vertex!(Vertex, position);

const SCREEN_QUAD: [Vertex; 6] = [
    Vertex::new(-1., -1.),
    Vertex::new(-1., 1.),
    Vertex::new(1., 1.),
    Vertex::new(-1., -1.),
    Vertex::new(1., -1.),
    Vertex::new(1., 1.),
];

type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn create_fullscreen_window(
    instance: Arc<Instance>,
    physical: PhysicalDevice,
) -> Result<(EventLoop<()>, Arc<Surface<Window>>, [u32; 2], Capabilities)> {
    let event_loop = EventLoop::new();

    let monitor_size = event_loop
        .primary_monitor().ok_or("no monitor found.")?.size();

    let surface = WindowBuilder::new()
        .with_inner_size(monitor_size)
        .with_min_inner_size(monitor_size)
        .with_fullscreen(Some(Fullscreen::Borderless(event_loop.primary_monitor())))
        .with_title("Mold Simulation")
        .build_vk_surface(&event_loop, instance)?;

    let dimensions = surface.window().inner_size().into();

    let caps = surface
        .capabilities(physical)
        .expect("failed to get surface capabilities");

    Ok((event_loop, surface, dimensions, caps))
}

fn main() -> Result<()> {
    let (args, _) = opts! {
        synopsis "wave-eq-sim - simulates the classical wave equation using rust + vulkan";
        opt pixel_size:f32=0.5, desc:"set the pixel size";
        opt actor_count:u32=100_000_000, desc: "number of actors";
    }
    .parse_or_exit();

    if std::fs::read_dir("./mold-pictures").is_err() {
        std::fs::create_dir("./mold-pictures").expect("failed to create picture folder");
    }

    let pixel_size: f32 = args.pixel_size;

    let actor_count = args.actor_count;

    let instance = {
        let extensions = vulkano_win::required_extensions();
        Instance::new(None, &extensions, None).expect("failed to create instance")
    };

    let physical = PhysicalDevice::enumerate(&instance)
        .next()
        .expect("no device available");

    dbg!(physical.name());

    let (events_loop, surface, dimensions, caps) =
        create_fullscreen_window(instance.clone(), physical)?;

    let queue_family = physical
        .queue_families()
        .find(|&q| q.supports_graphics() && surface.is_supported(q).unwrap_or(false))
        .expect("couldn't find a graphical queue family");

    let (device, mut queues) = Device::new(
        physical,
        &Features {
            shader_storage_image_extended_formats: true,
            ..Features::none()
        },
        &DeviceExtensions {
            khr_storage_buffer_storage_class: true,
            khr_swapchain: true,
            ..DeviceExtensions::none()
        },
        std::array::IntoIter::new([(queue_family, 0.5)]),
    )
    .expect("failed to create device");

    let queue = queues.next().ok_or("failed to get queue")?;

    let (mut swapchain, images) = {
        let alpha = caps
            .supported_composite_alpha
            .iter()
            .next()
            .ok_or("failed to get alpha channel")?;
        let format = caps.supported_formats[0].0;

        Swapchain::new(
            device.clone(),
            surface.clone(),
            caps.min_image_count.max(3),
            format,
            dimensions,
            1,
            ImageUsage::color_attachment(),
            &queue,
            SurfaceTransform::Identity,
            alpha,
            PresentMode::Immediate,
            FullscreenExclusive::Allowed,
            true,
            ColorSpace::SrgbNonLinear,
        )
        .expect("failed to create swapchain")
    };

    let vertex_buffer = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::all(),
        false,
        std::array::IntoIter::new(SCREEN_QUAD),
    )?;

    let actor_buffer = DeviceLocalBuffer::<[actors_cs::ty::Actor]>::array(
        device.clone(),
        args.actor_count as usize,
        BufferUsage {
            storage_buffer: true,
            ..BufferUsage::none()
        },
        vec![queue_family],
    )?;

    let phero_cs = phero_cs::Shader::load(device.clone()).expect("failed to create shader");
    let actors_cs = actors_cs::Shader::load(device.clone()).expect("failed to create shader");

    let cs_actors_uniform_buffer =
        CpuBufferPool::<actors_cs::ty::Data>::new(device.clone(), BufferUsage::all());

    let actors_compute_pipeline = Arc::new(
        ComputePipeline::new(device.clone(), &actors_cs.main_entry_point(), &(), None)
            .expect("failed to create compute pipeline"),
    );

    let phero_compute_pipeline = Arc::new(
        ComputePipeline::new(device.clone(), &phero_cs.main_entry_point(), &(), None)
            .expect("failed to create compute pipeline"),
    );

    let phero_map_dims = Dimensions::Dim2d {
        width: (dimensions[0] as f32 / pixel_size) as u32,
        height: (dimensions[1] as f32 / pixel_size) as u32,
    };

    dbg!(phero_map_dims);

    let phero_map_1 = StorageImage::with_usage(
        device.clone(),
        phero_map_dims,
        Format::R32G32Sfloat,
        ImageUsage {
            sampled: true,
            storage: true,
            transfer_destination: true,
            ..ImageUsage::none()
        },
        Some(queue.family()),
    )?;

    let phero_map_2 = StorageImage::with_usage(
        device.clone(),
        phero_map_dims,
        Format::R32G32Sfloat,
        ImageUsage {
            storage: true,
            transfer_destination: true,
            sampled: true,
            ..ImageUsage::none()
        },
        Some(queue.family()),
    )?;

    let sampler = Sampler::new(
        device.clone(),
        Filter::Nearest,
        Filter::Nearest,
        MipmapMode::Nearest,
        SamplerAddressMode::Repeat,
        SamplerAddressMode::Repeat,
        SamplerAddressMode::Repeat,
        0.0,
        1.0,
        0.0,
        0.0,
    )?;

    let vertex_shader = vs::Shader::load(device.clone()).expect("failed to create vertex shader");

    let vs_uniform_buffer = CpuBufferPool::<vs::ty::Data>::new(device.clone(), BufferUsage::all());

    let fragment_shader =
        fs::Shader::load(device.clone()).expect("failed to create fragment shader");

    let fs_uniform_buffer = CpuBufferPool::<fs::ty::Data>::new(device.clone(), BufferUsage::all());

    let render_pass = Arc::new(vulkano::ordered_passes_renderpass!(device.clone(),
        attachments: {
            color: {
                load: Clear,
                store: Store,
                format: swapchain.format(),
                samples: 1,
            }
        },
        passes: [
            {color: [color], depth_stencil: {}, input: []},
            {color: [color], depth_stencil: {}, input: []}
        ]
    )?);

    let pipeline = Arc::new(
        GraphicsPipeline::start()
            .vertex_input_single_buffer::<Vertex>()
            .vertex_shader(vertex_shader.main_entry_point(), ())
            .triangle_list()
            .viewports_dynamic_scissors_irrelevant(1)
            .fragment_shader(fragment_shader.main_entry_point(), ())
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device.clone())?,
    );

    let screenshot_render_pass = Arc::new(vulkano::single_pass_renderpass!(device.clone(),
        attachments: {
            color: {
                load: Clear,
                store: Store,
                format: Format::R8G8B8A8Unorm,
                samples: 1,
            }
        },
        pass: {color: [color], depth_stencil: {}}
    )?);

    let screenshot_pipeline = Arc::new(
        GraphicsPipeline::start()
            .vertex_input_single_buffer::<Vertex>()
            .vertex_shader(vertex_shader.main_entry_point(), ())
            .triangle_list()
            .viewports_dynamic_scissors_irrelevant(1)
            .fragment_shader(fragment_shader.main_entry_point(), ())
            .render_pass(Subpass::from(screenshot_render_pass.clone(), 0).unwrap())
            .build(device.clone())?,
    );

    let mut egui_platform = Platform::new(PlatformDescriptor {
        physical_width: dimensions[0],
        physical_height: dimensions[1],
        scale_factor: 1.,
        font_definitions: FontDefinitions::default(),
        style: Default::default(),
    });

    let mut egui_painter = egui_vulkano::Painter::new(
        device.clone(),
        queue.clone(),
        Subpass::from(render_pass.clone(), 1).unwrap(),
    )?;

    {
        let mut clear_builder =
            AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family())
                .unwrap();

        clear_builder
            .clear_color_image(phero_map_1.clone(), ClearValue::Float([0.0; 4]))
            .unwrap()
            .clear_color_image(phero_map_2.clone(), ClearValue::Float([0.0; 4]))
            .unwrap();

        let commands = clear_builder.build().unwrap();

        let _ = commands
            .execute(queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();
    }

    let mut dynamic_state = DynamicState::none();

    let mut framebuffers =
        window_size_dependent_setup(&images, render_pass.clone(), &mut dynamic_state);

    let mut recreate_swapchain = true;

    let mut previous_frame_end = Some(sync::now(device.clone()).boxed());

    let mut last_frame_time = Instant::now();
    let first_frame = Instant::now();

    let mut time = 0.;

    let mut mouse_pressed = false;

    let mut force_mult = 1.;

    let mut mouse_pos = [0., 0.];

    let mut clear_images = true;
    let mut render_image = phero_map_2;
    let mut back_image = phero_map_1;

    let mut diffusion_constant = 2.;
    let mut dissipation_constant = 27.;

    let mut sensor_angle: f32 = 70.;
    let mut sensor_distance: f32 = 5.;
    let mut actor_speed: f32 = 100.;
    let mut phero_strength: f32 = 20.;
    let mut turn_speed: f32 = 20.;
    let mut turn_gamma: f32 = 0.0;
    let mut randomness: f32 = 2.0;
    let mut init_gamma: f32 = 0.5;

    let mut hue: f32 = 0.;
    let mut gamma: f32 = 0.8;
    let mut brightness: f32 = 5.;

    let mut init_radius = 0.5;

    let mut relative_angle = 0.0;
    let mut random_angle = 360.;

    let mut actor_count = args.actor_count / 2;

    let mut zoom: f32 = 1.;

    let mut zoom_pos = [0.5, 0.5];

    let mut time_step = 0.01;

    let mut delta_time = 0.0;

    let mut take_screenshot = false;

    events_loop.run(move |event, _, control_flow| {
        egui_platform.handle_event(&event);
        if egui_platform.captures_event(&event) {
            return;
        }

        match event {
            Event::WindowEvent {
                event: winit::event::WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(_),
                ..
            } => {
                recreate_swapchain = true;
            }
            Event::WindowEvent {
                event:
                    WindowEvent::MouseWheel {
                        delta: MouseScrollDelta::LineDelta(_, y),
                        ..
                    },
                ..
            } => {
                zoom = (zoom * f32::powf(2., y as f32 / 8.)).max(1.);
            }
            Event::WindowEvent {
                event: WindowEvent::CursorMoved { position, .. },
                ..
            } => {
                mouse_pos = [
                    position.x as f32 / dimensions[0] as f32,
                    position.y as f32 / dimensions[1] as f32,
                ];
            }
            Event::WindowEvent {
                event:
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                virtual_keycode,
                                state,
                                ..
                            },
                        ..
                    },
                ..
            } => {
                if virtual_keycode == Some(VirtualKeyCode::R) && state == ElementState::Pressed {
                    clear_images = true;
                }

                if virtual_keycode == Some(VirtualKeyCode::Escape) {
                    *control_flow = ControlFlow::Exit;
                }
            }
            Event::RedrawEventsCleared => {
                previous_frame_end.as_mut().unwrap().cleanup_finished();

                delta_time = last_frame_time.elapsed().as_secs_f32();

                last_frame_time = Instant::now();

                {
                    let mut nav_delta = (0., 0.);

                    const ZONE_SIZE: f32 = 0.01;
                    const NAV_SPEED: f32 = 0.5;

                    if mouse_pos[0] < ZONE_SIZE {
                        nav_delta.0 -= 1.;
                    }

                    if mouse_pos[0] > 1. - ZONE_SIZE {
                        nav_delta.0 += 1.;
                    }

                    if mouse_pos[1] < ZONE_SIZE {
                        nav_delta.1 -= 1.;
                    }

                    if mouse_pos[1] > 1. - ZONE_SIZE {
                        nav_delta.1 += 1.;
                    }

                    zoom_pos[0] += nav_delta.0 * NAV_SPEED * delta_time / zoom;
                    zoom_pos[1] += nav_delta.1 * NAV_SPEED * delta_time / zoom
                        * dimensions[0] as f32
                        / dimensions[1] as f32;

                    zoom_pos[0] = zoom_pos[0].clamp(0.5 / zoom, 1. - 0.5 / zoom);
                    zoom_pos[1] = zoom_pos[1].clamp(0.5 / zoom, 1. - 0.5 / zoom);
                }

                if recreate_swapchain {
                    let dimensions: [u32; 2] = surface.window().inner_size().into();

                    let (new_swapchain, new_images) =
                        match swapchain.recreate_with_dimensions(dimensions) {
                            Ok(r) => r,
                            Err(SwapchainCreationError::UnsupportedDimensions) => return,
                            Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                        };

                    swapchain = new_swapchain;

                    framebuffers = window_size_dependent_setup(
                        &new_images,
                        render_pass.clone(),
                        &mut dynamic_state,
                    );

                    recreate_swapchain = false;
                }

                let (image_num, suboptimal, acquire_future) =
                    match swapchain::acquire_next_image(swapchain.clone(), None) {
                        Ok(r) => r,
                        Err(AcquireError::OutOfDate) => {
                            recreate_swapchain = true;
                            return;
                        }
                        Err(e) => panic!("Failed to acquire next image: {:?}", e),
                    };

                if suboptimal {
                    recreate_swapchain = true;
                }

                let clear_values = vec![[0.0, 0.0, 0.0, 1.0].into()];

                let phero_compute_layout = phero_compute_pipeline
                    .layout()
                    .descriptor_set_layout(0)
                    .ok_or("unable to get compute layout")
                    .unwrap();

                let actors_compute_layout = actors_compute_pipeline
                    .layout()
                    .descriptor_set_layout(0)
                    .ok_or("unable to get compute layout")
                    .unwrap();

                let fs_layout = pipeline.layout().descriptor_set_layout(0).unwrap();
                let vs_layout = pipeline.layout().descriptor_set_layout(1).unwrap();

                let phero_compute_uniforms = phero_cs::ty::PushConstantData {
                    delta_time: time_step,
                    init_image: clear_images as _,
                    diffusion_constant,
                    dissipation_constant,
                    time,
                };

                let fs_uniforms = fs_uniform_buffer
                    .next(fs::ty::Data {
                        hue,
                        gamma,
                        brightness,
                    })
                    .unwrap();

                let fs_set = Arc::new(
                    PersistentDescriptorSet::start(fs_layout.clone())
                        .add_sampled_image(render_image.clone(), sampler.clone())
                        .unwrap()
                        .add_buffer(fs_uniforms)
                        .unwrap()
                        .build()
                        .unwrap(),
                );

                let vs_uniforms = vs_uniform_buffer
                    .next(vs::ty::Data { zoom_pos, zoom })
                    .unwrap();

                let vs_set = Arc::new(
                    PersistentDescriptorSet::start(vs_layout.clone())
                        .add_buffer(vs_uniforms)
                        .unwrap()
                        .build()
                        .unwrap(),
                );

                let mut builder = AutoCommandBufferBuilder::primary_one_time_submit(
                    device.clone(),
                    queue.family(),
                )
                .unwrap();

                let actors_uniforms = cs_actors_uniform_buffer
                    .next(actors_cs::ty::Data {
                        actor_count,
                        delta_time: time_step,
                        time,
                        init: clear_images as _,
                        sensor_angle: sensor_angle / 360. * 2. * consts::PI,
                        sensor_distance,
                        actor_speed,
                        phero_strength,
                        turn_speed,
                        turn_gamma,
                        randomness,
                        init_radius,
                        relative_angle: relative_angle / 360. * 2. * consts::PI,
                        random_angle: random_angle / 360. * 2. * consts::PI,
                        init_gamma,
                    })
                    .unwrap();

                let actors_compute_set = Arc::new(
                    PersistentDescriptorSet::start(actors_compute_layout.clone())
                        .add_image(back_image.clone())
                        .unwrap()
                        .add_buffer(actor_buffer.clone())
                        .unwrap()
                        .add_buffer(actors_uniforms)
                        .unwrap()
                        .build()
                        .unwrap(),
                );

                const DISPATCH_SIZE: u32 = 65535;

                let workload = actor_count / 256 + 1;

                let dispatch_no = workload / DISPATCH_SIZE;

                for i in 0..dispatch_no {
                    builder
                    .dispatch(
                        [DISPATCH_SIZE, 1, 1],
                        actors_compute_pipeline.clone(),
                        actors_compute_set.clone(),
                        i * DISPATCH_SIZE,
                        vec![],
                    )
                    .unwrap();
                }

                builder
                .dispatch(
                    [workload % DISPATCH_SIZE, 1, 1],
                    actors_compute_pipeline.clone(),
                    actors_compute_set.clone(),
                    dispatch_no * DISPATCH_SIZE,
                    vec![],
                )
                .unwrap();

               

                let compute_set = Arc::new(
                    PersistentDescriptorSet::start(phero_compute_layout.clone())
                        .add_image(back_image.clone())
                        .unwrap()
                        .add_image(render_image.clone())
                        .unwrap()
                        .build()
                        .unwrap(),
                );
                builder
                    .dispatch(
                        [phero_map_dims.width() / 8, phero_map_dims.height() / 8, 1],
                        phero_compute_pipeline.clone(),
                        compute_set.clone(),
                        phero_compute_uniforms,
                        vec![],
                    )
                    .unwrap();

                mem::swap(&mut render_image, &mut back_image);

                builder
                    .begin_render_pass(
                        framebuffers[image_num].clone(),
                        SubpassContents::Inline,
                        clear_values,
                    )
                    .unwrap()
                    .draw(
                        pipeline.clone(),
                        &dynamic_state,
                        vertex_buffer.clone(),
                        (fs_set.clone(), vs_set.clone()),
                        (),
                        vec![],
                    )
                    .unwrap();

                egui_platform.begin_frame();

                egui::Window::new("Settings").show(&egui_platform.context(), |ui| {
                    ui.heading("General");

                    ui.add(
                        Slider::u32(&mut actor_count, 0..=args.actor_count)
                            .logarithmic(true)
                            .text("Actor Count"),
                    );

                    ui.advance_cursor(10.);

                    ui.heading("Pheromones");

                    ui.add(
                        Slider::f32(&mut diffusion_constant, 0.0..=10.0).text("Diffusion Constant"),
                    );

                    ui.add(
                        Slider::f32(&mut dissipation_constant, 0.0..=100.0)
                            .logarithmic(true)
                            .text("Dissipation Constant"),
                    );

                    ui.add(Slider::f32(&mut phero_strength, 0.0..=75.0).text("Strength"));

                    ui.advance_cursor(10.);

                    ui.heading("Actor Sensors");

                    ui.add(
                        Slider::f32(&mut sensor_angle, 0.0..=90.0)
                            .text("Angle")
                            .suffix("??"),
                    );
                    ui.add(Slider::f32(&mut sensor_distance, 1.0..=10.).text("Distance"));

                    ui.advance_cursor(10.);

                    ui.heading("Actor Movement");

                    ui.add(Slider::f32(&mut actor_speed, 10.0..=150.).text("Speed"));
                    ui.add(Slider::f32(&mut turn_speed, 0.0..=25.).text("Turn Speed"));
                    ui.add(Slider::f32(&mut turn_gamma, -1.0..=1.0).text("Turn Gamma"));
                    ui.add(Slider::f32(&mut randomness, 0.0..=10.).text("Randomness"));

                    ui.advance_cursor(10.);

                    ui.heading("Visual");

                    ui.add(Slider::f32(&mut hue, 0.0..=1.0).text("Hue"));
                    ui.add(Slider::f32(&mut gamma, 0.1..=1.4).text("Gamma"));
                    ui.add(Slider::f32(&mut brightness, 1.0..=20.0).text("Brightness"));

                    ui.advance_cursor(10.);

                    ui.heading("Initialization");

                    ui.add(Slider::f32(&mut init_radius, 0.0..=1.0).text("Radius"));
                    ui.add(Slider::f32(&mut init_gamma, 0.0..=2.0).text("Radial Distribution"));
                    ui.add(
                        Slider::f32(&mut relative_angle, 0.0..=360.)
                            .text("Relative Angle")
                            .suffix("??"),
                    );
                    ui.add(
                        Slider::f32(&mut random_angle, 0.0..=360.0)
                            .text("Random Angle")
                            .suffix("??"),
                    );

                    ui.advance_cursor(10.);

                    ui.heading("File");

                    take_screenshot = ui.button("Save Image").clicked();

                    ui.heading("Tips");

                    ui.label("Press R to reset the Simulation and apply initialization Settings!");
                });

                let (_output, clipped_shapes) = egui_platform.end_frame();

                egui_painter
                    .draw(
                        &mut builder,
                        &dynamic_state,
                        [dimensions[0] as f32, dimensions[1] as f32],
                        &egui_platform.context(),
                        clipped_shapes,
                    )
                    .unwrap();

                builder.end_render_pass().unwrap();

                let mut screenshot_buf = None;

                if take_screenshot {
                    let vs_uniforms = vs_uniform_buffer
                        .next(vs::ty::Data {
                            zoom_pos: [0.5, 0.5],
                            zoom: 1.,
                        })
                        .unwrap();

                    let vs_set = Arc::new(
                        PersistentDescriptorSet::start(vs_layout.clone())
                            .add_buffer(vs_uniforms)
                            .unwrap()
                            .build()
                            .unwrap(),
                    );

                    let screenshot_image = StorageImage::with_usage(
                        device.clone(),
                        phero_map_dims,
                        Format::R8G8B8A8Unorm,
                        ImageUsage {
                            transfer_source: true,
                            color_attachment: true,
                            ..ImageUsage::none()
                        },
                        Some(queue.family()),
                    )
                    .unwrap();

                    let buf = CpuAccessibleBuffer::from_iter(
                        device.clone(),
                        BufferUsage::all(),
                        false,
                        (0..phero_map_dims.width() * phero_map_dims.height() * 4).map(|_| 0u8),
                    )
                    .expect("failed to create buffer");

                    screenshot_buf = Some(buf.clone());

                    let screenshot_frame_buffer = Arc::new(
                        Framebuffer::start(screenshot_render_pass.clone())
                            .add(screenshot_image.clone())
                            .unwrap()
                            .build()
                            .unwrap(),
                    );

                    builder
                        .begin_render_pass(
                            screenshot_frame_buffer,
                            SubpassContents::Inline,
                            vec![[0.0; 4].into()],
                        )
                        .unwrap()
                        .draw(
                            screenshot_pipeline.clone(),
                            &DynamicState {
                                viewports: Some(vec![Viewport {
                                    origin: [0., 0.],
                                    dimensions: [
                                        phero_map_dims.width() as f32,
                                        phero_map_dims.height() as f32,
                                    ],
                                    depth_range: 0.0..1.0,
                                }]),
                                ..DynamicState::none()
                            },
                            vertex_buffer.clone(),
                            (fs_set, vs_set),
                            (),
                            vec![],
                        )
                        .unwrap()
                        .end_render_pass()
                        .unwrap()
                        .copy_image_to_buffer(screenshot_image, buf)
                        .unwrap();
                }

                let command_buffer = builder.build().unwrap();

                let future = previous_frame_end
                    .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_execute(queue.clone(), command_buffer)
                    .unwrap()
                    .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
                    .then_signal_fence_and_flush();

                time += time_step;

                match future {
                    Ok(future) => {
                        if take_screenshot {
                            future.wait(None).unwrap();
                            let buf = screenshot_buf.take().unwrap();
                            let buffer_content = buf.read().unwrap();

                            let image = image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(
                                phero_map_dims.width(),
                                phero_map_dims.height(),
                                &buffer_content[..],
                            )
                            .unwrap();

                            let local: chrono::DateTime<chrono::Local> = chrono::Local::now();

                            let filename =
                                format!("mold-pictures/mold-{}.png", local.timestamp_millis());

                            image.save(&filename).unwrap();
                        }

                        previous_frame_end = Some(future.boxed());
                    }
                    Err(FlushError::OutOfDate) => {
                        recreate_swapchain = true;
                        previous_frame_end = Some(sync::now(device.clone()).boxed());
                    }
                    Err(e) => {
                        println!("Failed to flush future: {:?}", e);
                        previous_frame_end = Some(sync::now(device.clone()).boxed());
                    }
                }

                clear_images = false;

                if delta_time < 1. / 60. {
                    std::thread::sleep(Duration::from_secs_f32(1. / 60. - delta_time));
                }
            }
            _ => (),
        }
    });
}

fn window_size_dependent_setup(
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
    dynamic_state: &mut DynamicState,
) -> Vec<Arc<dyn FramebufferAbstract + Send + Sync>> {
    let dimensions = images[0].dimensions();

    let viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [dimensions.width() as f32, dimensions.height() as f32],
        depth_range: 0.0..1.0,
    };
    dynamic_state.viewports = Some(vec![viewport]);

    images
        .iter()
        .map(|image| {
            Arc::new(
                Framebuffer::start(render_pass.clone())
                    .add(image.clone())
                    .unwrap()
                    .build()
                    .unwrap(),
            ) as Arc<dyn FramebufferAbstract + Send + Sync>
        })
        .collect::<Vec<_>>()
}
