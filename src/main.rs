use std::mem;
use std::sync::Arc;
use std::vec::Vec;
use std::{error::Error, time::Instant};

use rand::prelude::*;
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, CpuBufferPool},
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
    event::{ElementState, Event, KeyboardInput, MouseButton, VirtualKeyCode, WindowEvent},
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

const TIME_STEPS_PER_FRAME: u32 = 10;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn create_fullscreen_window(
    instance: Arc<Instance>,
    physical: PhysicalDevice,
) -> Result<(EventLoop<()>, Arc<Surface<Window>>, [u32; 2], Capabilities)> {
    let event_loop = EventLoop::new();

    let mode = event_loop
        .primary_monitor()
        .ok_or("no monitor found")?
        .video_modes()
        .max_by(|m1, m2| Ord::cmp(&m1.size().height, &m2.size().height))
        .ok_or("no resolution found")?;

    let surface = WindowBuilder::new()
        .with_inner_size(mode.size())
        .with_min_inner_size(mode.size())
        .with_fullscreen(Some(Fullscreen::Exclusive(mode)))
        .with_title("Wave Equation (Click and Drag to apply force to pixels)")
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
        opt pixel_size:u32=1, desc:"set the pixel size";
        opt actor_count:u32=500000, desc: "number of actors";
    }
    .parse_or_exit();

    let pixel_size: u32 = args.pixel_size;

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
            caps.min_image_count,
            format,
            dimensions,
            1,
            ImageUsage::color_attachment(),
            &queue,
            SurfaceTransform::Identity,
            alpha,
            PresentMode::Fifo,
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

    #[derive(Default)]
    struct Actor {
        position: [f32; 2],
        angle: f32,
        value: f32,
    }

    let actor_buffer = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage {
            storage_buffer: true,
            ..BufferUsage::none()
        },
        false,
        (0..actor_count).map(|_| Actor::default()),
    )?;

    let phero_cs = phero_cs::Shader::load(device.clone()).expect("failed to create shader");
    let actors_cs = actors_cs::Shader::load(device.clone()).expect("failed to create shader");

    let actors_compute_pipeline = Arc::new(
        ComputePipeline::new(device.clone(), &actors_cs.main_entry_point(), &(), None)
            .expect("failed to create compute pipeline"),
    );

    let phero_compute_pipeline = Arc::new(
        ComputePipeline::new(device.clone(), &phero_cs.main_entry_point(), &(), None)
            .expect("failed to create compute pipeline"),
    );

    let image1 = StorageImage::with_usage(
        device.clone(),
        Dimensions::Dim2d {
            width: dimensions[0] / pixel_size,
            height: dimensions[1] / pixel_size,
        },
        Format::R32G32Sfloat,
        ImageUsage {
            sampled: true,
            storage: true,
            transfer_destination: true,
            ..ImageUsage::none()
        },
        Some(queue.family()),
    )?;

    let image2 = StorageImage::with_usage(
        device.clone(),
        Dimensions::Dim2d {
            width: dimensions[0] / pixel_size,
            height: dimensions[1] / pixel_size,
        },
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
            .clear_color_image(image1.clone(), ClearValue::Float([0.0; 4]))
            .unwrap()
            .clear_color_image(image2.clone(), ClearValue::Float([0.0; 4]))
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
    let mut wave_pos = [0., 0.];

    let mut clear_images = true;
    let mut render_image = image2;
    let mut back_image = image1;

    let mut diffusion_constant = 10.;
    let mut dissipation_constant = 5.;

    let mut sensor_angle: f32 = std::f32::consts::PI / 3.;
    let mut sensor_distance: f32 = 2.;
    let mut sensor_size: i32 = 4;
    let mut actor_speed: f32 = 150.;
    let mut phero_strength: f32 = 20.;
    let mut turn_speed: f32 = 10.;
    let mut randomness: f32 = 2.5;

    let mut hue: f32 = 0.;
    let mut gamma: f32 = 0.8;
    let mut brightness: f32 = 5.;

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
                event: WindowEvent::MouseInput { state, button, .. },
                ..
            } => match state {
                ElementState::Pressed => {
                    mouse_pressed = true;
                    wave_pos = mouse_pos;

                    match button {
                        MouseButton::Left => force_mult = 1.,
                        MouseButton::Right => force_mult = -1.,
                        _ => (),
                    }
                }
                ElementState::Released => mouse_pressed = false,
            },
            Event::WindowEvent {
                event: WindowEvent::CursorMoved { position, .. },
                ..
            } => {
                mouse_pos = [
                    position.x as f32 / dimensions[0] as f32,
                    position.y as f32 / dimensions[1] as f32,
                ]
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

                let delta_time = 0.01;

                last_frame_time = Instant::now();

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

                let render_layout = pipeline.layout().descriptor_set_layout(0).unwrap();

                let phero_compute_uniforms = phero_cs::ty::PushConstantData {
                    delta_time: delta_time / TIME_STEPS_PER_FRAME as f32,
                    init_image: clear_images as _,
                    diffusion_constant,
                    dissipation_constant,
                    time,
                };

                let actors_compute_uniforms = actors_cs::ty::PushConstantData {
                    delta_time,
                    time,
                    init: clear_images as _,
                    sensor_angle,
                    sensor_distance,
                    sensor_size,
                    actor_speed,
                    phero_strength,
                    turn_speed,
                    randomness,
                };

                let render_uniforms = fs_uniform_buffer
                    .next(fs::ty::Data {
                        hue,
                        gamma,
                        brightness,
                    })
                    .unwrap();

                let render_set = Arc::new(
                    PersistentDescriptorSet::start(render_layout.clone())
                        .add_sampled_image(render_image.clone(), sampler.clone())
                        .unwrap()
                        .add_buffer(render_uniforms)
                        .unwrap()
                        .build()
                        .unwrap(),
                );

                let mut builder = AutoCommandBufferBuilder::primary_one_time_submit(
                    device.clone(),
                    queue.family(),
                )
                .unwrap();

                let actors_compute_set = Arc::new(
                    PersistentDescriptorSet::start(actors_compute_layout.clone())
                        .add_image(back_image.clone())
                        .unwrap()
                        .add_buffer(actor_buffer.clone())
                        .unwrap()
                        .build()
                        .unwrap(),
                );

                builder
                    .dispatch(
                        [actor_count / 32 + 1, 1, 1],
                        actors_compute_pipeline.clone(),
                        actors_compute_set.clone(),
                        actors_compute_uniforms,
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
                        [
                            dimensions[0] / pixel_size / 8 + 1,
                            dimensions[1] / pixel_size / 8 + 1,
                            1,
                        ],
                        phero_compute_pipeline.clone(),
                        compute_set.clone(),
                        phero_compute_uniforms,
                        vec![],
                    )
                    .unwrap();

                wave_pos[0] += (mouse_pos[0] - wave_pos[0]) * delta_time * 100.;
                wave_pos[1] += (mouse_pos[1] - wave_pos[1]) * delta_time * 100.;

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
                        render_set.clone(),
                        (),
                        vec![],
                    )
                    .unwrap();

                egui_platform.begin_frame();

                egui::Window::new("Settings").show(&egui_platform.context(), |ui| {
                    ui.heading("Pheromones");

                    ui.add(
                        Slider::f32(&mut diffusion_constant, 0.0..=20.0).text("Diffusion Constant"),
                    );

                    ui.add(
                        Slider::f32(&mut dissipation_constant, 0.0..=20.0)
                            .text("Dissipation Constant"),
                    );

                    ui.add(Slider::f32(&mut phero_strength, 0.0..=200.0).text("Strength"));

                    ui.heading("Actors");
                    ui.indent(1, |ui| {
                        ui.heading("Sensor");
                        ui.add(Slider::f32(&mut sensor_angle, 0.1..=3.141 / 2.).text("Angle"));
                        ui.add(Slider::f32(&mut sensor_distance, 1.0..=10.).text("Distance"));
                        ui.add(Slider::i32(&mut sensor_size, 1..=6).text("Size"));
                    });

                    ui.indent(2, |ui| {
                        ui.heading("Movement");
                        ui.add(Slider::f32(&mut actor_speed, 10.0..=500.).text("Speed"));
                        ui.add(Slider::f32(&mut turn_speed, 0.0..=200.).text("Turn Speed"));
                        ui.add(Slider::f32(&mut randomness, 0.0..=10.).text("Randomness"));
                    });

                    ui.heading("Visual");

                    ui.add(Slider::f32(&mut hue, 0.0..=1.0).text("Hue"));
                    ui.add(Slider::f32(&mut gamma, 0.1..=1.4).text("Gamma"));
                    ui.add(Slider::f32(&mut brightness, 1.0..=20.0).text("Brightness"));
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

                let command_buffer = builder.build().unwrap();

                let future = previous_frame_end
                    .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_execute(queue.clone(), command_buffer)
                    .unwrap()
                    .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
                    .then_signal_fence_and_flush();

                time += delta_time;

                match future {
                    Ok(future) => {
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
