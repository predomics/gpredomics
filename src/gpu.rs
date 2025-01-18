// src/gpu.rs
use std::collections::HashMap;
use std::os::raw::{c_int, c_float, c_void};

#[allow(improper_ctypes)]
extern "C" {
    fn create_kokkos_crs_matrix(
        nrows: c_int,
        ncols: c_int,
        nnz: c_int,
        row_map: *const c_int,
        col_idx: *const c_int,
        values:  *const c_float,
    ) -> *mut c_void;

    fn destroy_kokkos_crs_matrix(matrix_ptr: *mut c_void);

    fn spgemm_kokkos(matA: *mut c_void, matB: *mut c_void) -> *mut c_void;

    fn export_kokkos_crs_matrix(
        mat_ptr: *mut c_void,
        out_nrows: *mut c_int,
        out_ncols: *mut c_int,
        out_nnz:   *mut c_int,
        row_map_out: *mut c_int,
        col_idx_out: *mut c_int,
        vals_out:    *mut c_float
    ) -> c_int;
}

//------------------------------------------------------------------------------

/// A lightweight wrapper around the opaque Kokkos handle.
pub struct KokkosCrsMatrix {
    pub(crate) ptr: *mut c_void,
}

impl Drop for KokkosCrsMatrix {
    fn drop(&mut self) {
        unsafe {
            destroy_kokkos_crs_matrix(self.ptr);
        }
    }
}

//------------------------------------------------------------------------------

/// A plain Rust struct to hold the final CSR data on CPU.
#[derive(Debug)]
pub struct CsrMatrix {
    pub nrows: usize,
    pub ncols: usize,
    pub row_map: Vec<i32>,
    pub col_idx: Vec<i32>,
    pub values:  Vec<f32>,
}

//------------------------------------------------------------------------------

impl KokkosCrsMatrix {
    /// Build from a Rust HashMap<(r,c), double> by converting to CSR (float).
    pub fn new_from_hashmap(
        data: &HashMap<(usize, usize), f64>,
        nrows: usize,
        ncols: usize
    ) -> Self {
        let (row_map, col_idx, values) = build_csr(data, nrows, ncols);
        let nnz = col_idx.len() as i32;

        let ptr = unsafe {
            create_kokkos_crs_matrix(
                nrows as c_int,
                ncols as c_int,
                nnz,
                row_map.as_ptr(),
                col_idx.as_ptr(),
                values.as_ptr(),
            )
        };

        Self { ptr }
    }

    /// Perform spGEMM: C = self x other, returning new matrix handle.
    pub fn spgemm(&self, other: &KokkosCrsMatrix) -> KokkosCrsMatrix {
        let c_ptr = unsafe { spgemm_kokkos(self.ptr, other.ptr) };
        KokkosCrsMatrix { ptr: c_ptr }
    }

    /// Read back the CSR data from device into a local `CsrMatrix`.
    pub fn extract_csr(&self) -> CsrMatrix {
        let mut nrows = 0i32;
        let mut ncols = 0i32;
        let mut nnz   = 0i32;

        // First we call export to get shapes & also fill row_map/col_idx/vals.
        // But we need a second pass to allocate them properly once we know nnz,
        // unless we do a 2-call approach. Let's do a 2-call approach for clarity.

        // 1) do a "dummy" call with null pointers to get shape & nnz?
        // For simplicity, let's do it in one pass. We'll guess an upper bound or do a partial approach.
        // Alternatively, we can do a small helper function in C++ to just get shape/nnz first.
        // But let's illustrate one-pass with small data. We'll do a hack:
        // We'll call the function once to get shape & nnz only.

        unsafe {
            export_kokkos_crs_matrix(
                self.ptr,
                &mut nrows as *mut _,
                &mut ncols as *mut _,
                &mut nnz as *mut _,
                std::ptr::null_mut(), // row_map_out
                std::ptr::null_mut(), // col_idx_out
                std::ptr::null_mut(), // vals_out
            );
        }

        // Now allocate vectors of correct size
        let usize_nrows = nrows as usize;
        let usize_nnz   = nnz as usize;

        let mut row_map = vec![0i32; usize_nrows + 1];
        let mut col_idx = vec![0i32; usize_nnz];
        let mut values  = vec![0f32; usize_nnz];

        unsafe {
            export_kokkos_crs_matrix(
                self.ptr,
                &mut nrows as *mut _,
                &mut ncols as *mut _,
                &mut nnz as *mut _,
                row_map.as_mut_ptr(),
                col_idx.as_mut_ptr(),
                values.as_mut_ptr()
            );
        }

        CsrMatrix {
            nrows: nrows as usize,
            ncols: ncols as usize,
            row_map,
            col_idx,
            values,
        }
    }
}

//------------------------------------------------------------------------------

/// Convert HashMap<(r,c), f64> -> CSR arrays with float values.
fn build_csr(
    data: &HashMap<(usize, usize), f64>,
    nrows: usize,
    ncols: usize
) -> (Vec<i32>, Vec<i32>, Vec<f32>) {

    let mut row_entries = vec![Vec::<(i32,f32)>::new(); nrows];
    for (&(r,c), &val) in data.iter() {
        if r < nrows && c < ncols {
            row_entries[r].push((c as i32, val as f32));
        }
    }

    // row_map
    let mut row_map = vec![0i32; nrows+1];
    for r in 0..nrows {
        row_map[r+1] = row_map[r] + row_entries[r].len() as i32;
    }
    let nnz = row_map[nrows] as usize;

    // col_idx, values
    let mut col_idx = vec![0i32; nnz];
    let mut vals    = vec![0f32; nnz];
    for r in 0..nrows {
        let start = row_map[r] as usize;
        for (j, &(c_i, val_f)) in row_entries[r].iter().enumerate() {
            col_idx[start + j] = c_i;
            vals[start + j]    = val_f;
        }
    }

    (row_map, col_idx, vals)
}


/// Convert Vec<HashMap<usize,f64>> (a list of sparse column vectors) -> CSR arrays with float values.
fn build_csr_from_vec_as_cols(
    data: &Vec<HashMap<usize,f64>>,
    vec_size: usize
) -> (Vec<i32>, Vec<i32>, Vec<f32>) {
    let nrows = vec_size;
    let ncols = data.len();

    let mut row_entries = vec![Vec::<(i32,f32)>::new(); nrows];
    for (c, col) in data.iter().cloned().enumerate() {
        for (&r, &val) in col.iter(){
            if r < nrows && c < ncols {
                row_entries[r].push((c as i32, val as f32));
            }
        }
    }

    // row_map
    let mut row_map = vec![0i32; nrows+1];
    for r in 0..nrows {
        row_map[r+1] = row_map[r] + row_entries[r].len() as i32;
    }
    let nnz = row_map[nrows] as usize;

    // col_idx, values
    let mut col_idx = vec![0i32; nnz];
    let mut vals    = vec![0f32; nnz];
    for r in 0..nrows {
        let start = row_map[r] as usize;
        for (j, &(c_i, val_f)) in row_entries[r].iter().enumerate() {
            col_idx[start + j] = c_i;
            vals[start + j]    = val_f;
        }
    }

    (row_map, col_idx, vals)
}





//------------------------------------------------------------------------------

/// Example function demonstrating how to do a spSp GEMM and get results back.
pub fn example_spsp_gemm() -> CsrMatrix {
    use std::collections::HashMap;

    // Build A
    let mut Adata = HashMap::new();
    Adata.insert((0,0), 1.0);
    Adata.insert((0,2), 2.0);
    Adata.insert((2,1), 3.5);
    let A = KokkosCrsMatrix::new_from_hashmap(&Adata, 3, 4);

    // Build B
    let mut Bdata = HashMap::new();
    Bdata.insert((0,1), 4.0);
    Bdata.insert((1,3), 1.5);
    Bdata.insert((3,0), 2.2);
    let B = KokkosCrsMatrix::new_from_hashmap(&Bdata, 4, 5);

    // Multiply C = A x B
    let C = A.spgemm(&B);

    // Read result back to Rust
    let c_csr = C.extract_csr();
    c_csr
}
