use std::sync::Arc;
use std::vec::Vec;
use std::{error::Error, time::Instant};
use std::{f32::consts, mem};

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
        .with_fullscreen(Some(Fullscreen::Borderless(event_loop.primary_monitor())))
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
        opt pixel_size:f32=0.5, desc:"set the pixel size";
        opt actor_count:u32=1000000, desc: "number of actors";
    }
    .parse_or_exit();

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

    let actor_buffer = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage {
            storage_buffer: true,
            ..BufferUsage::none()
        },
        false,
        (0..actor_count).map(|_| actors_cs::ty::Actor {
            angle: 0.,
            position: [0., 0.],
            _dummy0: [0; 4],
        }),
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
    let mut sensor_size: i32 = 4;
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
                    delta_time: time_step / TIME_STEPS_PER_FRAME as f32,
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
                        sensor_size,
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

                builder
                    .dispatch(
                        [actor_count / 64 + 1, 1, 1],
                        actors_compute_pipeline.clone(),
                        actors_compute_set.clone(),
                        (),
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
                            phero_map_dims.width() / 8 + 1,
                            phero_map_dims.height() / 8 + 1,
                            1,
                        ],
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
                    ui.indent(1, |ui| {
                        ui.add(
                            Slider::u32(&mut actor_count, 0..=args.actor_count)
                                .logarithmic(true)
                                .text("Actor Count"),
                        );
                    });

                    ui.advance_cursor(10.);

                    ui.heading("Pheromones");

                    ui.indent(1, |ui| {
                        ui.add(
                            Slider::f32(&mut diffusion_constant, 0.0..=20.0)
                                .text("Diffusion Constant"),
                        );

                        ui.add(
                            Slider::f32(&mut dissipation_constant, 0.0..=20.0)
                                .text("Dissipation Constant"),
                        );

                        ui.add(Slider::f32(&mut phero_strength, 0.0..=200.0).text("Strength"));
                    });

                    ui.advance_cursor(10.);

                    ui.heading("Actor Sensors");

                    ui.indent(1, |ui| {
                        ui.add(
                            Slider::f32(&mut sensor_angle, 15.0..=90.0)
                                .text("Angle")
                                .suffix("°"),
                        );
                        ui.add(Slider::f32(&mut sensor_distance, 1.0..=10.).text("Distance"));
                        ui.add(Slider::i32(&mut sensor_size, 1..=6).text("Size"));
                    });

                    ui.advance_cursor(10.);

                    ui.heading("Actor Movement");

                    ui.indent(2, |ui| {
                        ui.add(Slider::f32(&mut actor_speed, 10.0..=150.).text("Speed"));
                        ui.add(Slider::f32(&mut turn_speed, 0.0..=100.).text("Turn Speed"));
                        ui.add(Slider::f32(&mut turn_gamma, -2.0..=2.0).text("Turn Gamma"));
                        ui.add(Slider::f32(&mut randomness, 0.0..=10.).text("Randomness"));
                    });

                    ui.advance_cursor(10.);

                    ui.heading("Visual");

                    ui.indent(3, |ui| {
                        ui.add(Slider::f32(&mut hue, 0.0..=1.0).text("Hue"));
                        ui.add(Slider::f32(&mut gamma, 0.1..=1.4).text("Gamma"));
                        ui.add(Slider::f32(&mut brightness, 1.0..=20.0).text("Brightness"));
                    });

                    ui.advance_cursor(10.);

                    ui.heading("Initialization");

                    ui.indent(3, |ui| {
                        ui.add(Slider::f32(&mut init_radius, 0.0..=1.0).text("Radius"));
                        ui.add(Slider::f32(&mut init_gamma, 0.0..=2.0).text("Radial Distribution"));
                        ui.add(
                            Slider::f32(&mut relative_angle, 0.0..=360.)
                                .text("Relative Angle")
                                .suffix("°"),
                        );
                        ui.add(
                            Slider::f32(&mut random_angle, 0.0..=360.0)
                                .text("Random Angle")
                                .suffix("°"),
                        );
                    });

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
