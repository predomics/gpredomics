// 1) Define a tile size constant
const TILE_SIZE: u32 = 16;
const TILE_ELEMENTS: u32 = TILE_SIZE * TILE_SIZE;
const MATRIX_SIZE_U32: u32 = 5000;

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;

// 2) We'll allocate workgroup (shared) memory for a 16×16 tile from A and B.
//    Each tile is T x T = 16 x 16 = 256 floats.
var<workgroup> tileA: array<f32, TILE_ELEMENTS>;
var<workgroup> tileB: array<f32, TILE_ELEMENTS>;

@compute @workgroup_size(TILE_SIZE, TILE_SIZE, 1)
fn main(
    @builtin(global_invocation_id) global_id : vec3<u32>,
    @builtin(local_invocation_id) local_id : vec3<u32>,
    @builtin(workgroup_id) workgroup_id : vec3<u32>
) {
    // Matrix dimension (we'll pass it via push_constants or a special uniform if needed).
    // For a static example, you can bake it in or do an @override constant.
    let N = MATRIX_SIZE_U32;

    // Row/column in the final output
    let row = global_id.y;
    let col = global_id.x;

    // Local row/col within a single 16×16 workgroup tile
    let localRow = local_id.y;
    let localCol = local_id.x;

    // We accumulate the dot product in `acc`.
    var acc: f32 = 0.0;

    // Number of tiles we'll need in the K dimension
    // Example: for an NxN multiply, we break the K dimension into tile blocks
    let numTiles = (N + TILE_SIZE - 1u) / TILE_SIZE; // round up if not multiple of 16

    // We'll loop over each tile along the K dimension
    for (var t = 0u; t < numTiles; t = t + 1u) {
        // 3) Load a sub-block (tile) from A and B into shared memory
        //    We must check bounds because 5000 is not necessarily multiple of 16.

        // Column index in A to load from
        let kA = t * TILE_SIZE + localCol;
        // Row index in B to load from
        let kB = t * TILE_SIZE + localRow;

        // If row < N && kA < N, load from A, else 0
        if (row < N && kA < N) {
            tileA[localRow * TILE_SIZE + localCol] = A[(row * N) + kA];
        } else {
            tileA[localRow * TILE_SIZE + localCol] = 0.0;
        }

        // If kB < N && col < N, load from B, else 0
        if (kB < N && col < N) {
            tileB[localRow * TILE_SIZE + localCol] = B[(kB * N) + col];
        } else {
            tileB[localRow * TILE_SIZE + localCol] = 0.0;
        }

        // 4) Wait for every thread to finish loading tile data before we use it
        workgroupBarrier();

        // 5) Compute partial sums within the tile
        //    We'll do a standard dot product across the dimension = TILE_SIZE
        for (var i = 0u; i < TILE_SIZE; i = i + 1u) {
            acc = acc + tileA[localRow * TILE_SIZE + i] * tileB[i * TILE_SIZE + localCol];
        }

        // 6) Make sure we’re done using the tile before overwriting it in next iteration
        workgroupBarrier();
    }

    // 7) Write the accumulated sum back to C, if within valid range
    if row < N && col < N {
        C[row * N + col] = acc;
    }
}