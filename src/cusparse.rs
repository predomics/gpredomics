use std::collections::HashMap;

use std::ffi::c_void;
use std::ptr::null_mut;


/// Suppose `hashmap` maps (row, col) -> value, for nonzero entries.
/// shape: (n_rows, n_cols).
fn hashmap_to_csr(
    hashmap: &HashMap<(usize, usize), f64>,
    n_rows: usize,
    n_cols: usize,
) -> (Vec<i32>, Vec<i32>, Vec<f32>) {
    // 1) We'll build a structure like:
    //    rowPtr: length = n_rows + 1
    //    colIdx, values: length = nnz = number of nonzero entries in the HashMap
    //
    //    We'll store them as i32 indices + f32 values for the GPU (common usage),
    //    but you could do i64 + f64 if needed.
    
    // We need to gather entries per row.
    // A quick approach: build a Vec<Vec<(col, val)>> for each row.
    let mut row_entries = vec![Vec::<(usize, f64)>::new(); n_rows];
    for (&(r, c), &val) in hashmap.iter() {
        if r >= n_rows || c >= n_cols {
            // Handle out-of-bounds if necessary, or ignore
            continue;
        }
        row_entries[r].push((c, val));
    }

    // 2) Now we know how many nonzeros in each row.
    //    We can build rowPtr by prefix sum.
    let mut row_ptr = vec![0i32; n_rows + 1];
    for r in 0..n_rows {
        row_ptr[r + 1] = row_ptr[r] + row_entries[r].len() as i32;
    }

    let nnz = row_ptr[n_rows] as usize;
    let mut col_idx = vec![0i32; nnz];
    let mut values  = vec![0f32; nnz];

    // 3) Fill col_idx and values by row
    for r in 0..n_rows {
        let start = row_ptr[r] as usize;
        for (offset, &(c, val)) in row_entries[r].iter().enumerate() {
            col_idx[start + offset] = c as i32;
            values[start + offset]  = val as f32; // or do any transform (log, etc.)
        }
    }

    (row_ptr, col_idx, values)
}

// Suppose we define these somewhere:
extern "C" {
    fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, size: usize, kind: i32) -> i32;
    // etc.
}

// Some constants
const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1; // not the real value, just example

fn upload_csr_to_gpu(
    row_ptr: &[i32],
    col_idx: &[i32],
    values:  &[f32],
) -> (/*d_rowptr*/ *mut c_void,
      /*d_colidx*/ *mut c_void,
      /*d_values*/ *mut c_void)
{
    let nnz = values.len();

    // row_ptr has length n_rows+1
    let size_rowptr_bytes = row_ptr.len() * std::mem::size_of::<i32>();
    let size_colidx_bytes = col_idx.len() * std::mem::size_of::<i32>();
    let size_values_bytes = values.len()   * std::mem::size_of::<f32>();

    let mut d_rowptr: *mut c_void = null_mut();
    let mut d_colidx: *mut c_void = null_mut();
    let mut d_values: *mut c_void = null_mut();

    unsafe {
        cudaMalloc(&mut d_rowptr, size_rowptr_bytes);
        cudaMalloc(&mut d_colidx, size_colidx_bytes);
        cudaMalloc(&mut d_values, size_values_bytes);

        cudaMemcpy(d_rowptr, row_ptr.as_ptr() as *const c_void,
                   size_rowptr_bytes, CUDA_MEMCPY_HOST_TO_DEVICE);
        cudaMemcpy(d_colidx, col_idx.as_ptr() as *const c_void,
                   size_colidx_bytes, CUDA_MEMCPY_HOST_TO_DEVICE);
        cudaMemcpy(d_values, values.as_ptr() as *const c_void,
                   size_values_bytes, CUDA_MEMCPY_HOST_TO_DEVICE);
    }

    (d_rowptr, d_colidx, d_values)
}

use std::ptr::null_mut;
use std::os::raw::c_void;

// Suppose you have the FFI from some `cusparse_sys` module:
extern "C" {
    fn cusparseSpMM_bufferSize(
        handle: cusparseHandle_t,
        opA:    cusparseOperation_t,
        opB:    cusparseOperation_t,
        alpha:  *const c_void,
        matA:   cusparseSpMatDescr_t,
        matB:   cusparseDnMatDescr_t,
        beta:   *const c_void,
        matC:   cusparseDnMatDescr_t,
        computeType: cudaDataType,
        alg:    cusparseSpMMAlg_t,
        bufferSize: *mut usize
    ) -> cusparseStatus_t;

    fn cusparseSpMM(
        handle: cusparseHandle_t,
        opA:    cusparseOperation_t,
        opB:    cusparseOperation_t,
        alpha:  *const c_void,
        matA:   cusparseSpMatDescr_t,
        matB:   cusparseDnMatDescr_t,
        beta:   *const c_void,
        matC:   cusparseDnMatDescr_t,
        computeType: cudaDataType,
        alg:    cusparseSpMMAlg_t,
        externalBuffer: *mut c_void
    ) -> cusparseStatus_t;

    // etc. for creation of descriptors, etc.
}


