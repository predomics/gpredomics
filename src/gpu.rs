use std::time::Instant;
use wgpu::util::DeviceExt;

const MATRIX_SIZE: usize = 5000; // Size of the matrix (MATRIX_SIZE x MATRIX_SIZE)



pub fn main() {

    let shader_source = include_str!("shader.wgsl");
    


    println!("Starting gpu main");
    // Initialize wgpu
    pollster::block_on(async {
        let instance = wgpu::Instance::default();
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .expect("Failed to find an adapter");

        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor::default(), None)
            .await
            .expect("Failed to create device");

        // Matrix data
        let a_data: Vec<f32> = vec![1.0; MATRIX_SIZE * MATRIX_SIZE];
        let b_data: Vec<f32> = vec![1.0; MATRIX_SIZE * MATRIX_SIZE];
        let mut c_data: Vec<f32> = vec![0.0; MATRIX_SIZE * MATRIX_SIZE];

        // Buffers
        let a_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("A Buffer"),
            contents: bytemuck::cast_slice(&a_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let b_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("B Buffer"),
            contents: bytemuck::cast_slice(&b_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let c_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("C Buffer"),
            size: (MATRIX_SIZE * MATRIX_SIZE * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // -- Create a staging buffer used for reading results back to the CPU.
        let c_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("C Staging Buffer"),
            size: (MATRIX_SIZE * MATRIX_SIZE * std::mem::size_of::<f32>()) as u64,
            // Must include MAP_READ so we can map it on the CPU side
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Shader module
        //let shader_code = SHADER.replace("${MATRIX_SIZE_U32}", &MATRIX_SIZE.to_string());
        //let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        //    label: Some("Matrix Multiplication Shader"),
        //    source: wgpu::ShaderSource::Wgsl(shader_code.into()),
        //});
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("My Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // -- NEW CODE: Create a BindGroupLayout that matches the shader's bindings.
        //    We need binding=0,1,2, each with the correct type.
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Compute Bind Group Layout"),
            entries: &[
                // @group(0) @binding(0): read-only storage buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @group(0) @binding(1): read-only storage buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @group(0) @binding(2): read-write storage buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });


        // -- NEW CODE: Create a BindGroup that holds our a_buffer, b_buffer, and c_buffer.
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(a_buffer.as_entire_buffer_binding()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(b_buffer.as_entire_buffer_binding()),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(c_buffer.as_entire_buffer_binding()),
                },
            ],
        });

        // -- NEW CODE: Create a PipelineLayout that uses our bind group layout.
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None
        });


        // Benchmark
        let iterations = 100;
        let start_time = Instant::now();

        for i in 0..iterations {
            // Submit work to GPU
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Command Encoder"),
            });

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Compute Pass"),
                    timestamp_writes: None,
                });
                pass.set_bind_group(0, &bind_group, &[]);
                pass.set_pipeline(&pipeline);
                pass.dispatch_workgroups(
                    (MATRIX_SIZE as u32 + 16 - 1) / 16,
                    (MATRIX_SIZE as u32 + 16 - 1) / 16,
                    1
                );
                
            }

            
            // -- Now copy the result from c_buffer into the staging buffer:
            encoder.copy_buffer_to_buffer(
                &c_buffer,         // src
                0,
                &c_staging_buffer, // dst
                0,
                (MATRIX_SIZE * MATRIX_SIZE * std::mem::size_of::<f32>()) as u64,
            );

            queue.submit(Some(encoder.finish()));

            // -- Wait for the GPU to finish
            // (pollster's block_on is usually enough, but we can do an explicit poll):
            device.poll(wgpu::Maintain::Wait);

            // -- Map staging buffer to CPU
            let buffer_slice = c_staging_buffer.slice(..);
            buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
            device.poll(wgpu::Maintain::Wait);

            // -- Access the data
            let data = buffer_slice.get_mapped_range();
            let result_slice: &[f32] = bytemuck::cast_slice(&data);

            // If you want, copy into c_data
            c_data.copy_from_slice(result_slice);

            // Unmap so we can write next iteration
            drop(data);
            c_staging_buffer.unmap();

            // For example, check the first element
            if i == 0 {
                println!("First iteration, c_data[0..4] = {:?}", &c_data[0..4]);
            }
        }

        let elapsed = start_time.elapsed();
        println!(
            "Elapsed time for {} iterations: {:.2?}",
            iterations, elapsed
        );
        println!("Average time per iteration: {:.4?} ms", elapsed.as_secs_f64() / iterations as f64 * 1000.0);
    });
}
