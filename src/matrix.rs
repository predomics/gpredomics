use tensorflow::{Tensor, SparseTensor};
use std::collections::HashMap;

fn create_sparse_tensor_x(
    x_map: HashMap<(u8, u8), f32>,
    nrows: u64,
    ncols: u64,
) -> tensorflow::Result<SparseTensor> {
    let mut indices = vec![];
    let mut values = vec![];

    for ((row, col), value) in x_map {
        indices.push(row as i64);
        indices.push(col as i64);
        values.push(value);
    }

    let indices_tensor = Tensor::new(&[(indices.len() / 2) as u64, 2])
        .with_values(&indices)?;
    let values_tensor = Tensor::new(&[values.len() as u64])
        .with_values(&values)?;
    let shape_tensor = Tensor::new(&[2])
        .with_values(&[nrows as i64, ncols as i64])?;

    SparseTensor::new(indices_tensor, values_tensor, shape_tensor)
}

fn create_sparse_tensor_im(
    models: Vec<HashMap<usize, i8>>,
    nfeatures: u64,
) -> tensorflow::Result<SparseTensor> {
    let mut indices = vec![];
    let mut values = vec![];
    let nmodels = models.len();

    for (model_index, model) in models.iter().enumerate() {
        for (&feature_index, &value) in model {
            indices.push(feature_index as i64);
            indices.push(model_index as i64);
            values.push(value as f32);
        }
    }

    let indices_tensor = Tensor::new(&[(indices.len() / 2) as u64, 2])
        .with_values(&indices)?;
    let values_tensor = Tensor::new(&[values.len() as u64])
        .with_values(&values)?;
    let shape_tensor = Tensor::new(&[2])
        .with_values(&[nfeatures as i64, nmodels as i64])?;

    SparseTensor::new(indices_tensor, values_tensor, shape_tensor)
}

fn multiply_sparse_matrices(
    x_sparse: SparseTensor,
    im_sparse: SparseTensor,
) -> tensorflow::Result<Tensor<f32>> {
    let mut graph = tensorflow::Graph::new();
    let x = graph.sparse_placeholder();
    let im = graph.sparse_placeholder();

    // Sparse matrix multiplication
    let is = graph.sparse_tensor_dense_matmul(x, im)?;

    // Create a session to run the graph
    let session = tensorflow::Session::new(&tensorflow::SessionOptions::new(), &graph)?;

    let mut step = tensorflow::SessionRunArgs::new();
    step.add_feed(&x, 0, &x_sparse);
    step.add_feed(&im, 0, &im_sparse);

    // Request the result of multiplication
    let result_token = step.request_fetch(&is, 0);
    session.run(&mut step)?;

    // Fetch the result
    step.fetch(result_token)
}

use std::collections::HashMap;

fn main() -> tensorflow::Result<()> {
    // Example input data
    let mut x_map = HashMap::new();
    x_map.insert((0, 0), 1.0);
    x_map.insert((0, 1), 2.0);
    x_map.insert((1, 0), 3.0);

    let nrows = 2;
    let ncols = 2;
    let x_sparse = create_sparse_tensor_x(x_map, nrows, ncols)?;

    let models = vec![
        HashMap::from([(0, 1), (1, -1)]),
        HashMap::from([(0, 2), (1, 3)]),
    ];
    let im_sparse = create_sparse_tensor_im(models, ncols)?;

    let is_dense = multiply_sparse_matrices(x_sparse, im_sparse)?;

    println!("Score Matrix: {:?}", is_dense);
    Ok(())
}