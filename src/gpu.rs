use std::time::Instant;
use wgpu::util::DeviceExt;

const MATRIX_SIZE: usize = 1024; // Size of the matrix (MATRIX_SIZE x MATRIX_SIZE)

// Shader for matrix multiplication
const SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;

    if (row < ${MATRIX_SIZE_U32}u && col < ${MATRIX_SIZE_U32}u) {
        var sum: f32 = 0.0;
        for (var k: u32 = 0u; k < ${MATRIX_SIZE_U32}u; k = k + 1u) {
            let a_val = a[row * ${MATRIX_SIZE_U32}u + k];
            let b_val = b[k * ${MATRIX_SIZE_U32}u + col];
            sum = sum + a_val * b_val;
        }
        c[row * ${MATRIX_SIZE_U32}u + col] = sum;
    }
}
"#;

pub fn main() {

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

        // Shader module
        let shader_code = SHADER.replace("${MATRIX_SIZE_U32}", &MATRIX_SIZE.to_string());
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Matrix Multiplication Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_code.into()),
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

        for _ in 0..iterations {
            // Submit work to GPU
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Command Encoder"),
            });

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Compute Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline);
            }

            queue.submit(Some(encoder.finish()));
        }

        let elapsed = start_time.elapsed();
        println!(
            "Elapsed time for {} iterations: {:.2?}",
            iterations, elapsed
        );
        println!("Average time per iteration: {:.4?} ms", elapsed.as_secs_f64() / iterations as f64 * 1000.0);
    });
}
