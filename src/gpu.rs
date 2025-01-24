use std::time::Instant;
use wgpu::util::DeviceExt;
use std::collections::HashMap;
use bytemuck;

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

    println!("csr matrix");
    for r in 0..rows {
        let mut i: usize=row_ptr[r] as usize;
        println!("{}: {}",r,
            (0..row_ptr[r+1]-row_ptr[r]).map( |c| {
            if c<col_idx[i] {
                ".".to_string()
            }
            else {
                i+=1;
                val[i-1].to_string()
            } 
        }).collect::<Vec<String>>().join(" "));
    }

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
    _pad: u32
    // possibly some padding
}

pub fn gpu_eval(X: &HashMap<(usize,usize),f64>, models: &Vec<HashMap<usize,i8>>, feature_selection: &HashMap<usize,usize>, samples: usize) {

    let spgemm_source = include_str!("spgemm.wgsl");
    let instance = wgpu::Instance::default();
    let model_nb = models.len();

    // preparing X (features) representation in CSR
    let (row_ptrX, col_idxX, valX) = hashmap_to_csr(X, feature_selection, samples); 

    // preparing MM (model matrix) representation in CSC
    let (col_ptrMM, row_idxMM, valMM ) = vechash_to_csc(models, feature_selection);

    
    pollster::block_on(async {
        println!("Starting gpu main");
        // Initialize wgpu

        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .expect("Failed to find an adapter");

        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor::default(), None)
            .await
            .expect("Failed to create device");

        // X buffer for X in CSR
        let row_ptrX_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("row_ptrX"),
            contents: bytemuck::cast_slice(&row_ptrX),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let col_idxX_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("col_idxX"),
            contents: bytemuck::cast_slice(&col_idxX),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let valX_buf     = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("valX"),
            contents: bytemuck::cast_slice(&valX),
            usage: wgpu::BufferUsages::STORAGE,
        });
        
        // Score Matrix (SM) 
        let SM_size = (samples*model_nb*std::mem::size_of::<f32>()) as u64;
        let mut SM: Vec<f32> = vec![0.0; samples*model_nb];

        let SM_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SM Buffer"),
            size: SM_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC,  // if you want to read it back
            mapped_at_creation: false,
        });

        // -- Create a staging buffer used for reading results back to the CPU.
        let SM_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SM Staging Buffer"),
            size: SM_size,
            // Must include MAP_READ so we can map it on the CPU side
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params_data = MatrixMultParams { 
                R: samples as u32,
                C: feature_selection.len() as u32,
                N: model_nb as u32,
                _pad: 0
            };
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Params Buffer"),
            contents: bytemuck::bytes_of(&params_data),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        
        // 5) Build the pipeline layout, referencing 8 total bindings (A in CSR, B in CSC, C, uniform).
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SpGEMM Bind Group Layout"),
            entries: &[
                // Binding 0 => row_ptrA_buf (STORAGE, read-only)
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
                // Binding 1 => col_idxA_buf
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
                // Binding 2 => valA_buf
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
                // Binding 3 => col_ptrB_buf
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
                // Binding 4 => row_idxB_buf
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
                // Binding 5 => valB_buf
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
                // Binding 6 => c_buf (read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 7 => params_buf (uniform)
                        // 7) params -> uniform buffer (MatrixMultParams struct)
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }
            ],
        });

        // Shader module
        //let shader_code = SHADER.replace("${MATRIX_SIZE_U32}", &MATRIX_SIZE.to_string());
        //let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        //    label: Some("Matrix Multiplication Shader"),
        //    source: wgpu::ShaderSource::Wgsl(shader_code.into()),
        //});
        let spgemm_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SpGEMM shadder"),
            source: wgpu::ShaderSource::Wgsl(spgemm_source.into()),
        });


        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SpGEMM Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SpGEMM Pipeline"),
            layout: Some(&pipeline_layout),
            module: &spgemm_shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None
        });

        // create buffer for MM
        let col_ptrMM_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("row_ptrX"),
            contents: bytemuck::cast_slice(&col_ptrMM),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let row_idxMM_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("col_idxX"),
            contents: bytemuck::cast_slice(&row_idxMM),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let valMM_buf     = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("valX"),
            contents: bytemuck::cast_slice(&valMM),
            usage: wgpu::BufferUsages::STORAGE,
        });


        // -- NEW CODE: Create a BindGroup that holds our a_buffer, b_buffer, and c_buffer.
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                // Binding 0 => row_ptrA_buf (STORAGE, read-only)
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(row_ptrX_buf.as_entire_buffer_binding()),
                },
                // Binding 1 => col_idxA_buf
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(col_idxX_buf.as_entire_buffer_binding()),
                },
                // Binding 2 => valA_buf
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(valX_buf.as_entire_buffer_binding()),
                },
                // Binding 3 => col_ptrB_buf
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(col_ptrMM_buf.as_entire_buffer_binding()),
                },
                // Binding 4 => row_idxB_buf
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Buffer(row_idxMM_buf.as_entire_buffer_binding()),
                },
                // Binding 5 => valB_buf
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Buffer(valMM_buf.as_entire_buffer_binding()),
                },
                // Binding 6 => c_buf (read_write)
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Buffer(SM_buf.as_entire_buffer_binding()),
                },
                // Binding 7 => params_buf (uniform)
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::Buffer(params_buf.as_entire_buffer_binding()),
                }
            ],
        });

        // -- NEW CODE: Create a PipelineLayout that uses our bind group layout.
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
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
                //pass.dispatch_workgroups(
                //    (samples as u32 + 15) / 16,
                //    (feature_selection.len() as u32 + 15) / 16,
                //    1
                //);
                // We want a thread for each cell of SM => (S, M)
                let group_x = (model_nb as u32 + 15) / 16; // S Sample are x (rows)
                let group_y = (samples as u32 + 15) / 16; // M Model are y (columns)
                pass.dispatch_workgroups(group_x, group_y, 1);
            }

            //let mut SM: Vec<Vec<f32>> = Vec::new();
            
            // -- Now copy the result from c_buffer into the staging buffer:
            encoder.copy_buffer_to_buffer(
                &SM_buf,         // src
                0,
                &SM_staging_buffer, // dst
                0,
                SM_size,
            );

            queue.submit(Some(encoder.finish()));

            // -- Wait for the GPU to finish
            // (pollster's block_on is usually enough, but we can do an explicit poll):
            device.poll(wgpu::Maintain::Wait);

            // -- Map staging buffer to CPU
            let buffer_slice = SM_staging_buffer.slice(..);
            buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
            device.poll(wgpu::Maintain::Wait);

            // -- Access the data
            let data = buffer_slice.get_mapped_range();
            let result_slice: &[f32] = bytemuck::cast_slice(&data);

            // If you want, copy into c_data
            SM.copy_from_slice(result_slice);

            // Unmap so we can write next iteration
            drop(data);
            SM_staging_buffer.unmap();

            // For example, check the first element
            if i == 0 {
                println!("First iteration, score matrix");
                for row in 0..samples {
                    println!(
                        "{}: {}", row, (0..model_nb).map(|c| {SM[row * model_nb + c].to_string()}).collect::<Vec<String>>().join(" ")
                    );
                }
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



//#[cfg(test)]
//mod tests {
//    use super::*;
//    use std::collections::HashMap;
//
//    #[test]
//    fn test_hash_to_csr_rectangular() {
//        // Example: a 3x5 matrix (R=3, C=5)
//        // mat =
//        // [ 0   0   7.0  0   0  ]
//        // [ 4.2 0   0    0   2.2]
//        // [ 0   0   0    0   1.0]
//
//        let R = 3;
//        let C = 5;
//        let mut mat_map = HashMap::new();
//        mat_map.insert((0, 2), 7.0);
//        mat_map.insert((1, 0), 4.2);
//        mat_map.insert((1, 4), 2.2);
//        mat_map.insert((2, 4), 1.0);
//
//        let (row_ptr, col_idx, val) = hash_to_csr(R, C, &mat_map);
//
//        // row_counts = [1, 2, 1]
//        // => row_ptr = [0, 1, 3, 4] => total nnz = 4
//
//        assert_eq!(row_ptr, vec![0, 1, 3, 4]);
//
//        // col_idx / val:
//        // row 0 => indices [0..1): col=2 => val=7.0
//        // row 1 => indices [1..3): col=? => val=?  (two entries)
//        // row 2 => indices [3..4): col=4 => val=1.0
//        //
//        // The exact order in row 1 depends on insertion order, but we'll do a basic check
//        // for correctness.
//        assert_eq!(col_idx.len(), 4);
//        assert_eq!(val.len(), 4);
//
//        // row 0 => col_idx[0]=2 => val[0]=7.0
//        assert_eq!(col_idx[0], 2);
//        assert!((val[0] - 7.0).abs() < 1e-6);
//
//        // row 1 => col_idx[1..3] => {0, 4}, val => {4.2, 2.2}
//        // row 2 => col_idx[3] => {4}, val => {1.0}
//    }
//}