use bincode::config;
use wgpu::util::DeviceExt;
use wgpu::{BindGroupEntry, BindingResource, CommandEncoderDescriptor, ComputePassDescriptor};
use std::collections::HashMap;
use bytemuck;
use crate::individual::{Individual, RATIO_LANG,RAW_TYPE,LOG_TYPE,PREVALENCE_TYPE};
use crate::param::{GpuMemoryPolicy, GPU};
use log::warn;

/// Convert a HashMap<(row, col), f64> to CSR format for an R x C matrix. 
///
/// - `mat_map`: A mapping from (row, col) -> value, representing nonzero entries.
/// - `row_selection` : A mapping initial row -> index representing a selection of rows
///
/// Returns (row_ptr, col_idx, val):
///   - `row_ptr` has length R+1.
///   - `col_idx` and `val` each have length = total number of nonzeros.
///   - For row r, the nonzero entries are in the index range [row_ptr[r] .. row_ptr[r+1])
///     of col_idx/val.
///
/// Indices are stored as `u32`, and values as `f32`.
///
/// Note: This function does no checks for out-of-bounds row/col in `mat_map`. If you want
///       to ensure row < R and col < C, add boundary checks as needed.
fn hashmap_to_csr(
    mat_map: &HashMap<(usize, usize), f64>,
    column_selection: &HashMap<usize,usize>,
    rows: usize
) -> (Vec<u32>, Vec<u32>, Vec<f32>) {
    // 1) Count nonzeros in each row
    //    row_counts[r] = how many entries in row r
    //let R=row_selection.len();
    let mut row_counts = vec![0u32; rows];

    for (r, raw_c) in mat_map.keys() {
        // Optional boundary check (comment out if not desired):
        // if r >= R || c >= C {
        //     panic!("(r, c) = ({}, {}) out of bounds for R={}, C={}", r, c, R, C);
        // }
        if column_selection.contains_key(raw_c) {
            row_counts[*r] += 1;
        }
    }
    //println!("row count {:?}",row_counts);

    // 2) Build row_ptr via prefix sums. row_ptr[r+1] = row_ptr[r] + row_counts[r].
    //    row_ptr[R] will hold the total number of nonzeros across all rows.
    let mut row_ptr = vec![0u32; rows + 1];
    for r in 0..rows {
        row_ptr[r + 1] = row_ptr[r] + row_counts[r];
    }
    let nnz = row_ptr[rows] as usize; // total nonzeros

    // 3) Allocate col_idx and val arrays
    let mut col_idx = vec![0u32; nnz];
    let mut val = vec![0f32; nnz];

    // 4) We'll track the "next free index" per row in a separate array
    //    (or we can reuse row_ptr after we copy it).
    let mut next_free = row_ptr.clone();

    // 5) Fill col_idx/val
    for (&(r, raw_c), &value_f64) in mat_map.iter() {
        if let Some(&c) = column_selection.get(&raw_c) {
            let insert_pos  = next_free[r] as u32;
            col_idx[insert_pos as usize] = c as u32;
            val[insert_pos as usize] = value_f64 as f32;
            next_free[r] += 1;
        }
    }

    // row_ptr[r] and row_ptr[r+1] define the sub-range in col_idx/val
    // for row r. We sort col_idx and val in tandem, by ascending col_idx.
    for r in 0..rows {
        let start = row_ptr[r] as usize;
        let end   = row_ptr[r + 1] as usize;

        // We want to sort col_idx[start..end] and val[start..end]
        // together, using col_idx as the key.

        // Easiest approach: zip them into a temp slice, sort by the first field
        let mut zipped = col_idx[start..end]
            .iter()
            .copied()
            .zip(val[start..end].iter().copied())
            .collect::<Vec<_>>();

        // Sort by col_idx (the first of the pair)
        zipped.sort_by_key(|&(cidx, _)| cidx);

        // Write back
        for i in 0..zipped.len() {
            col_idx[start + i] = zipped[i].0;
            val[start + i]     = zipped[i].1;
        }
    }

    //println!("csr matrix");
    //for r in 0..rows {
    //    let mut i: usize=row_ptr[r] as usize;
    //    println!("{}: {}",r,
    //        (0..row_ptr[r+1]-row_ptr[r]).map( |c| {
    //        if c<col_idx[i] {
    //            ".".to_string()
    //        }
    //        else {
    //            i+=1;
    //            val[i-1].to_string()
    //        } 
    //    }).collect::<Vec<String>>().join(" "));
    //}

    //println!("row ptr {:?}\ncol idx {:?}\nval {:?}", row_ptr, col_idx, val);
    (row_ptr, col_idx, val)
}


/// Convert a vector of hashmaps representing columns of a matrix B
/// into CSC (Compressed Sparse Column) format with f32 values.
/// 
/// - `mat_cols`: length = number of columns, so column `c` is b_cols[c].
/// - Each column is a HashMap<row_u8, value_f64>.
/// - `R`: number of rows in B (optional boundary check).
///
/// Returns (col_ptr, row_idx, val):
///   - `col_ptr`: length = num_cols + 1
///   - `row_idx`, `val`: each length = total number of nonzeros
///   - For column c, the nonzero entries are in index range [col_ptr[c] .. col_ptr[c+1]) 
///     of `row_idx`/`val`.
fn vechash_to_csc(
    mat_cols: &Vec<HashMap<usize, i8>>,
    row_selection: &HashMap<usize,usize>,
) -> (Vec<u32>, Vec<u32>, Vec<f32>) {
    let num_cols = mat_cols.len();

    // 1) Count total nonzeros per column
    //    col_ptrB[c+1] = col_ptrB[c] + number_of_nonzeros_in_col
    let mut col_ptr = vec![0u32; num_cols + 1];
    for (c, col_map) in mat_cols.iter().enumerate() {
        col_ptr[c + 1] = col_ptr[c] + (col_map.len() as u32);
    }

    // total nonzeros
    let nnz = col_ptr[num_cols] as usize;

    // 2) Allocate row_idxB and valB
    let mut row_idx = vec![0u32; nnz];
    let mut val     = vec![0f32; nnz];

    // 3) We'll track the "next free insertion index" for each column
    //    (initially col_ptrB, which we will clone)
    let mut next_free = col_ptr.clone();

    // 4) Fill row_idxB and valB
    for (c, col_map) in mat_cols.iter().enumerate() {
        for (&raw_row, &val_f64) in col_map.iter() {
            // optional boundary check
            //if (row_u8 as usize) >= R {
            //    panic!("Row index {} out of bounds for {} rows", row_u8, R);
            //}
            let row_u32 = row_selection[&raw_row] as u32;

            // insertion position for this column
            let pos = next_free[c];
            row_idx[pos as usize] = row_u32;
            val[pos as usize]     = val_f64 as f32;

            next_free[c] += 1;
        }
    }

    // col_ptr[c] and col_ptr[c+1] define the sub-range in row_idx/val
    // for col c. We sort row_idx and val in tandem, by ascending row_idx.
    for c in 0..num_cols {
        let start = col_ptr[c] as usize;
        let end   = col_ptr[c + 1] as usize;

        // We want to sort col_idx[start..end] and val[start..end]
        // together, using col_idx as the key.

        // Easiest approach: zip them into a temp slice, sort by the first field
        let mut zipped = row_idx[start..end]
            .iter()
            .copied()
            .zip(val[start..end].iter().copied())
            .collect::<Vec<_>>();

        // Sort by col_idx (the first of the pair)
        zipped.sort_by_key(|&(ridx, _)| ridx);

        // Write back
        for i in 0..zipped.len() {
            row_idx[start + i] = zipped[i].0;
            val[start + i]     = zipped[i].1;
        }
    }

    //println!("col ptr {:?}\nrow idx {:?}\nval {:?}", col_ptr, row_idx, val);
    (col_ptr, row_idx, val)
}


/// An object to store the size of matrix
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MatrixMultParams {
    R: u32,
    C: u32,
    N: u32,
    threshold: f32,
    epsilon: f32
    // possibly some padding
}

pub struct GpuAssay {
    // WGPU core
    pub config: GPU,
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,

    // Pipeline and layout
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
 
    // Buffers for the X matrix (CSR)
    row_ptrX_buf: wgpu::Buffer,
    col_idxX_buf: wgpu::Buffer,
    valX_buf: wgpu::Buffer,

    // Dimensions
    samples: usize,
    feature_count: usize,
    feature_selection: HashMap<usize,usize>,

    // The WGSL code (could also be a static string or loaded from file)
    shader_module: wgpu::ShaderModule,
 
    // Reusable staging buffer for SM readback
    // We'll create it with some maximum size if we know it, or create on the fly
    max_sm_size: usize,
    sm_staging_buf: wgpu::Buffer,
    sm_buf: wgpu::Buffer,
    sm_size_bytes: u64
}

impl GpuAssay {

    /// Synchronous constructor that internally does async wgpu setup
    pub fn new(
        x_map: &HashMap<(usize, usize), f64>,
        feature_selection: &Vec<usize>,
        samples: usize,
        max_model_nb: usize,
        config: &GPU
    ) -> Self {
        // Just call the async constructor in a pollster::block_on
        pollster::block_on(Self::new_async(x_map, feature_selection, samples,max_model_nb, config))
    }

    pub fn log_memory_status(&self) {
        log::debug!(
            "GPU Memory Policy: {:?} | Buffer: {}/{}MB | Total: {}/{}MB",
            self.config.memory_policy,
            self.device.limits().max_storage_buffer_binding_size / 1024 / 1024,
            self.config.max_buffer_size_mb,
            self.device.limits().max_buffer_size / 1024 / 1024,
            self.config.max_total_memory_mb
        );
    }

    pub async fn new_async(
        x_map: &HashMap<(usize, usize), f64>,
        feature_selection: &Vec<usize>,
        samples: usize,
        max_model_nb: usize,
        config: &GPU,
    ) -> Self {
        
        // 1) Build wgpu
        let instance = wgpu::Instance::default();

        // First, try to get an adapter without forcing CPU fallback
        let adapter_result = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }).await;

        // Now we can match on the result
        let adapter = match adapter_result {
            Some(adapter) => {
                // Control device type
                let info = adapter.get_info();
                
                if !config.fallback_to_cpu && info.device_type == wgpu::DeviceType::Cpu {
                    panic!("No compatible graphics card detected. The program requires a graphics card compatible with WGPU (Vulkan, Metal, DX12 or WebGPU) as the fallback_to_cpu option is disabled.");
                }
                
                log::info!("\x1b[0;32mGPU adapter selected: {} ({:?})\x1b[0m", info.name, info.device_type);
                adapter
            },
            None => {
                if !config.fallback_to_cpu {
                    panic!("No graphics adapter could be initialised and the fallback_to_cpu option is disabled.");
                }
                
                // Try explicitly with CPU fallback
                log::warn!("No compatible GPU found, trying with CPU fallback");
                instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::LowPower,
                    force_fallback_adapter: true,
                    compatible_surface: None,
                }).await.expect("Unable to initialise even a spare CPU adapter")
            }
        };

        // Get material constraints
        // Perplexity said that a match should return the same type, let's see. 
        let hardware_limits = adapter.limits();
        let requested_total_size = config.max_total_memory_mb.saturating_mul(1024).saturating_mul(1024);
        let requested_buffer_size = config.max_buffer_size_mb.saturating_mul(1024).saturating_mul(1024);

        // Calcul des limites selon la politique choisie
        let (final_total, final_buffer) = match config.memory_policy {
            GpuMemoryPolicy::Strict => {
                if requested_total_size > hardware_limits.max_buffer_size {
                    panic!(
                        "GPU Strict policy: requested total memory ({} MB) exceeds hardware limit ({} MB)",
                        config.max_total_memory_mb,
                        hardware_limits.max_buffer_size / (1024 * 1024)
                    );
                } 
                if requested_buffer_size > hardware_limits.max_storage_buffer_binding_size {
                    panic!(
                        "GPU Strict policy: requested buffer size ({} MB) exceeds hardware limit ({} MB)",
                        config.max_buffer_size_mb,
                        hardware_limits.max_storage_buffer_binding_size / (1024 * 1024)
                    );
                }
                (requested_total_size, requested_buffer_size)
            },

            GpuMemoryPolicy::Adaptive => (
                requested_total_size.min(hardware_limits.max_buffer_size),
                requested_buffer_size.min(hardware_limits.max_storage_buffer_binding_size)
            ),
            
            GpuMemoryPolicy::Performance => (
                hardware_limits.max_buffer_size,
                hardware_limits.max_storage_buffer_binding_size
            ),
        };

        let required_limits = wgpu::Limits {
            max_storage_buffer_binding_size: final_buffer,
            max_buffer_size: final_total,
            max_storage_buffers_per_shader_stage: 8,
            ..wgpu::Limits::downlevel_defaults()
        };


        let (device, queue) = match adapter.request_device(&wgpu::DeviceDescriptor {required_limits,..Default::default()}, None)
        .await {
            Ok(d) => d,
            Err(e) if config.fallback_to_cpu => {
                log::warn!("GPU initialization failed: {}. Falling back to CPU", e);
                let cpu_adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
                    force_fallback_adapter: true,
                    ..Default::default()
                })
                .await
                .expect("Failed to create CPU fallback");
                cpu_adapter.request_device(&wgpu::DeviceDescriptor::default(), None)
                    .await
                    .expect("CPU fallback failed")
            }
            Err(e) => panic!("GPU initialization failed: {}", e),
        };

        let feature_map: HashMap<usize,usize> = feature_selection.iter().enumerate().map(|(index,feature)| {(*feature,index)}).collect();

        // 2) Create CSR for X
        let (row_ptrX, col_idxX, valX) = hashmap_to_csr(x_map, &feature_map, samples);

        // 3) Create the buffers for X
        let row_ptrX_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("row_ptrX Buf"),
            contents: bytemuck::cast_slice(&row_ptrX),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let col_idxX_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("col_idxX Buf"),
            contents: bytemuck::cast_slice(&col_idxX),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let valX_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("valX Buf"),
            contents: bytemuck::cast_slice(&valX),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // 4) Create the Shader
        let spgemm_source = include_str!("spgemm.wgsl");
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SpGEMM shadder"),
            source: wgpu::ShaderSource::Wgsl(spgemm_source.into()),
        });

        // 5) Create Bind Group Layout, Pipeline Layout, and Pipeline
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SpGEMM BGL"),
            entries: &[
                // 0 => row_ptrX
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
                // 1 => col_idxX
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
                // 2 => valX
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 3 => col_ptrMM
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 4 => row_idxMM
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 5 => valMM
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 6 => valMM
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 7 => SM
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 8 => Params
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SpGEMM Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SpGEMM Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None
        });

        // 6) Create a staging buffer big enough for the maximum SM we expect.
        // For demonstration, let's pick something big or simply 4 * samples * some max model count
        // or rely on dynamic creation each time.
        let max_sm_size = samples * max_model_nb;
        let sm_staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SM Staging Buf"),
            size: (max_sm_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // 2) Create an SM buffer of size (samples * num_models)
        let sm_count = samples * max_model_nb;
        let sm_size_bytes = (sm_count * std::mem::size_of::<f32>()) as u64;
        let sm_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SM Buffer"),
            size: sm_size_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        Self {
            config: config.clone(),
            instance,
            adapter,
            device,
            queue,
            pipeline,
            bind_group_layout,
            pipeline_layout,
            row_ptrX_buf,
            col_idxX_buf,
            valX_buf,
            samples,
            feature_count: feature_selection.len(),
            feature_selection: feature_map,
            shader_module,
            max_sm_size,
            sm_staging_buf,
            sm_buf,
            sm_size_bytes
            }
    }

    pub fn compute_scores(
        &self,
        models: &Vec<Individual>,
        threshold: f32
    ) -> Vec<f32> {
        let num_models = models.len();
        let (col_ptrMM, row_idxMM, valMM) = vechash_to_csc(
            &models.iter().map(|i| {i.features.clone()}).collect(), 
            &self.feature_selection);

        // 1) Create GPU buffers for MM
        let col_ptrMM_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("col_ptrMM_buf"),
            contents: bytemuck::cast_slice(&col_ptrMM),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let row_idxMM_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("row_idxMM_buf"),
            contents: bytemuck::cast_slice(&row_idxMM),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let valMM_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("valMM_buf"),
            contents: bytemuck::cast_slice(&valMM),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let dataType_buf= self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dataType_buf"),
            contents: bytemuck::cast_slice(&models.iter().map(|i| {
                match (i.data_type,i.language) {
                    (RAW_TYPE, RATIO_LANG) => 3,
                    (LOG_TYPE, RATIO_LANG) => 4,
                    (PREVALENCE_TYPE, RATIO_LANG) => 5,
                    (LOG_TYPE, _) => 1,
                    (PREVALENCE_TYPE, _) => 2,
                    _ => 0
                }
            }).collect::<Vec<u32>>()),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // 3) Create the uniform params buffer
        let params_data = MatrixMultParams {
            R: self.samples as u32,
            C: self.feature_count as u32,
            N: num_models as u32,
            threshold,
            epsilon: models[0].epsilon as f32,
        };

        let params_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Params Buf"),
            contents: bytemuck::bytes_of(&params_data),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // 4) Build the bind group for this run
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.bind_group_layout,
            entries: &[
                // X
                BindGroupEntry { binding: 0, resource: BindingResource::Buffer(self.row_ptrX_buf.as_entire_buffer_binding()) },
                BindGroupEntry { binding: 1, resource: BindingResource::Buffer(self.col_idxX_buf.as_entire_buffer_binding()) },
                BindGroupEntry { binding: 2, resource: BindingResource::Buffer(self.valX_buf.as_entire_buffer_binding()) },
                // MM
                BindGroupEntry { binding: 3, resource: BindingResource::Buffer(col_ptrMM_buf.as_entire_buffer_binding()) },
                BindGroupEntry { binding: 4, resource: BindingResource::Buffer(row_idxMM_buf.as_entire_buffer_binding()) },
                BindGroupEntry { binding: 5, resource: BindingResource::Buffer(valMM_buf.as_entire_buffer_binding()) },
                BindGroupEntry { binding: 6, resource: BindingResource::Buffer(dataType_buf.as_entire_buffer_binding()) },
                // SM
                BindGroupEntry { binding: 7, resource: BindingResource::Buffer(self.sm_buf.as_entire_buffer_binding()) },
                // Params
                BindGroupEntry { binding: 8, resource: BindingResource::Buffer(params_buf.as_entire_buffer_binding()) },
            ],
            label: Some("Compute Bind Group for MM"),
        });

        // 5) Dispatch
        let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("ComputeScore Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("ComputeScore Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // each cell => (samples, num_models)
            let group_x = (num_models as u32 + 15) / 16;
            let group_y = (self.samples as u32 + 15) / 16;
            pass.dispatch_workgroups(group_x, group_y, 1);
        }

       // 6) Copy from sm_buf to staging
        encoder.copy_buffer_to_buffer(
            &self.sm_buf, 
            0,
            &self.sm_staging_buf, 
            0,
            self.sm_size_bytes
        );

        self.queue.submit(Some(encoder.finish()));

        // Wait for the GPU to complete
        self.device.poll(wgpu::Maintain::Wait);

        // 7) Map + read
        {
            let sm_count = self.samples * num_models;
            let buffer_size: u64 = (sm_count * std::mem::size_of::<f32>()) as u64;
            let slice = self.sm_staging_buf.slice(0..buffer_size);

            // Start map request in read mode
            slice.map_async(wgpu::MapMode::Read, |_| {});

            // Block until complete
            self.device.poll(wgpu::Maintain::Wait);

            // Now read the data
            let data = slice.get_mapped_range();
            let scores: &[f32] = bytemuck::cast_slice(&data);

            let mut result_vec = vec![0f32; sm_count];
            result_vec.copy_from_slice(scores);

            drop(data);
            self.sm_staging_buf.unmap();

            // 'result_vec' now has the final SM
            return result_vec;
        }

    }

    pub fn get_max_buffer_size(config: &GPU) -> u32 {
        pollster::block_on(async {
            let instance = wgpu::Instance::default();
            let adapter_future = instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            });

            let adapter = match adapter_future.await {
                Some(a) => a,
                None => {
                    if !config.fallback_to_cpu {
                        return 134217728 / 2;
                    }

                    match instance.request_adapter(&wgpu::RequestAdapterOptions {
                        power_preference: wgpu::PowerPreference::LowPower,
                        force_fallback_adapter: true,
                        compatible_surface: None,
                    }).await {
                        Some(a) => a,
                        None => return 134217728 / 2,
                    }
                }
            };

            let hardware_limits = adapter.limits();
            let config_max_size = config.max_buffer_size_mb.saturating_mul(1024 * 1024);

            match config.memory_policy {
                GpuMemoryPolicy::Strict => {
                    let max_size = config_max_size.min(hardware_limits.max_storage_buffer_binding_size);
                    if config_max_size > hardware_limits.max_storage_buffer_binding_size {
                        panic!(
                            "GPU Strict policy: requested total memory ({} MB) exceeds hardware limit ({} MB)",
                            config.max_buffer_size_mb,
                            hardware_limits.max_storage_buffer_binding_size / (1024 * 1024)
                        );
                    }
                    max_size
                }
                GpuMemoryPolicy::Adaptive => {
                    let max_size = hardware_limits.max_storage_buffer_binding_size.min(config_max_size);
                    if config_max_size > hardware_limits.max_storage_buffer_binding_size {
                        warn!(
                            "The value of param.gpu.max_buffer_size_mb ({} MB) is too high and will be truncated to the hardware limit ({} MB).",
                            config.max_buffer_size_mb,
                            hardware_limits.max_storage_buffer_binding_size / (1024 * 1024)
                        );
                    }
                    max_size
                }
                GpuMemoryPolicy::Performance => {
                    if config_max_size > hardware_limits.max_storage_buffer_binding_size {
                        warn!(
                            "The value of param.gpu.max_buffer_size_mb ({} MB) is too high and will be ignored in favor of the hardware limit ({} MB).",
                            config.max_buffer_size_mb,
                            hardware_limits.max_storage_buffer_binding_size / (1024 * 1024)
                        );
                    }
                    hardware_limits.max_storage_buffer_binding_size
                }
            }
        })
    }
}
