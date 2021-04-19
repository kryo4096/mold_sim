use std::{error::Error, sync::Arc};

use vulkano::{
    buffer::CpuBufferPool,
    descriptor::{descriptor_set::PersistentDescriptorSet, DescriptorSet, PipelineLayoutAbstract},
    device::Device,
    pipeline::{
        shader::{ComputeEntryPoint, EntryPointAbstract},
        ComputePipeline, ComputePipelineAbstract,
    },
};

struct ComputeShader<U> {
    device: Arc<Device>,
    pipeline: Arc<dyn ComputePipelineAbstract>,
    sets: Vec<Arc<dyn DescriptorSet>>,
    uniform_buffer: Arc<CpuBufferPool<U>>,
}

impl<U> ComputeShader<U> {
    fn create<S, BIter>(
        device: Arc<Device>,
        shader: S,
        specialization: S::SpecializationConstants,
        buffers: BIter,
    ) -> Result<Self, Box<dyn Error>>
    where
        S: EntryPointAbstract,
        S::SpecializationConstants: Clone,
        S::PipelineLayout: Clone,
        BIter: IntoIterator,
        BIter::Item: vulkano::buffer::BufferAccess,
    {
        let pipeline = ComputePipeline::new(device.clone(), &shader, &specialization, None)?;

        let mut i = 0;

        while let Some(set_layout) = pipeline.descriptor_set_layout(i) {
            
        }

        let set = PersistentDescriptorSet::start();
    }
}
