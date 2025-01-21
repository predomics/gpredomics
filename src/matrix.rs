use arrayfire::{self as af, Dim4, MatProp, SparseFormat};
use std::collections::HashMap;

fn main() {
    // Example sparse matrix as a HashMap
    let sparse_matrix: HashMap<(usize, usize), f32> = vec![
        ((0, 0), 1.0),
        ((0, 1), 2.0),
        ((1, 2), 3.0),
    ]
    .into_iter()
    .collect();

    // Example dense matrix as Vec<HashMap<usize, i8>>
    let dense_matrix: Vec<HashMap<usize, i8>> = vec![
        [(0, 1), (1, 2)].into_iter().collect(), // Column 0
        [(1, 3)].into_iter().collect(),         // Column 1
        [(0, 4), (2, 5)].into_iter().collect(), // Column 2
    ];

    // Convert sparse HashMap to ArrayFire sparse matrix
    let af_sparse = hashmap_to_af_sparse(&sparse_matrix, 2, 3); // Assuming 2x3 matrix

    // Convert dense Vec<HashMap<usize, i8>> to ArrayFire dense matrix
    let af_dense = vec_hashmap_to_af_dense(&dense_matrix, 3, 3); // Assuming 3x3 matrix

    // Perform sparse x dense multiplication
    let result = af::sparse_matmul(&af_sparse, &af_dense, MatProp::NONE);

    // Convert ArrayFire matrix back to Vec<Vec<f32>>
    let result_vec = af_to_vec_vec(&result);

    println!("{:?}", result_vec);
}

// Helper function: Convert sparse HashMap to ArrayFire sparse matrix
fn hashmap_to_af_sparse(
    hashmap: &HashMap<(usize, usize), f32>,
    rows: u64,
    cols: u64,
) -> af::SparseArray<f32> {
    let mut row_indices = Vec::new();
    let mut col_indices = Vec::new();
    let mut values = Vec::new();

    for (&(row, col), &value) in hashmap {
        row_indices.push(row as u32);
        col_indices.push(col as u32);
        values.push(value);
    }

    let row_indices = af::Array::new(&row_indices, Dim4::new(&[row_indices.len() as u64, 1, 1, 1]));
    let col_indices = af::Array::new(&col_indices, Dim4::new(&[col_indices.len() as u64, 1, 1, 1]));
    let values = af::Array::new(&values, Dim4::new(&[values.len() as u64, 1, 1, 1]));

    af::create_sparse_array_from_triplets(&values, &row_indices, &col_indices, rows, cols, SparseFormat::CSR)
}

// Helper function: Convert Vec<HashMap<usize, i8>> to ArrayFire dense matrix
fn vec_hashmap_to_af_dense(
    vec: &Vec<HashMap<usize, i8>>,
    rows: u64,
    cols: u64,
) -> af::Array<f32> {
    let mut dense_matrix = vec![0.0; (rows * cols) as usize];

    for (col_idx, col_map) in vec.iter().enumerate() {
        for (&row_idx, &value) in col_map {
            dense_matrix[(row_idx * cols as usize + col_idx) as usize] = value as f32;
        }
    }

    af::Array::new(&dense_matrix, Dim4::new(&[rows, cols, 1, 1]))
}

// Helper function: Convert ArrayFire dense matrix to Vec<Vec<f32>>
fn af_to_vec_vec(matrix: &af::Array<f32>) -> Vec<Vec<f32>> {
    let dims = matrix.dims();
    let rows = dims[0] as usize;
    let cols = dims[1] as usize;

    let mut host_data = vec![0.0; rows * cols];
    matrix.host(&mut host_data);

    (0..rows)
        .map(|row| (0..cols).map(|col| host_data[row * cols + col]).collect())
        .collect()
}
