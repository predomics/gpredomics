// Multiply X (S x F) by MM (F x M) => SM (S x M).

struct SpGemmParams {
    S: u32,     // #samples
    F: u32,     // #features
    M: u32,     // #models
    _pad: u32  // 16-byte alignment
};

// Bind group layout (indices) assumption:
//   0 -> row_ptrX  (array<u32>)  // CSR for Xtrans
//   1 -> col_idxX  (array<u32>)
//   2 -> valX      (array<f32>)
//
//   3 -> col_ptrMM (array<u32>)  // CSC for MM
//   4 -> row_idxMM (array<u32>)
//   5 -> valMM     (array<f32>)
//
//   6 -> SM        (array<f32>)  // Output, size = S*M
//   7 -> params    (uniform SpGemmParams)

@group(0) @binding(0) var<storage, read>  row_ptrX : array<u32>;
@group(0) @binding(1) var<storage, read>  col_idxX : array<u32>;
@group(0) @binding(2) var<storage, read>  valX     : array<f32>;

@group(0) @binding(3) var<storage, read>  col_ptrMM : array<u32>;
@group(0) @binding(4) var<storage, read>  row_idxMM : array<u32>;
@group(0) @binding(5) var<storage, read>  valMM     : array<f32>;

@group(0) @binding(6) var<storage, read_write> SM : array<f32>;

@group(0) @binding(7) var<uniform> params : SpGemmParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;  // 0..S
    let col = gid.x;  // 0..M

    let S = params.S;
    let F = params.F;
    let M = params.M;

    // Out-of-bounds => skip
    if (row >= S || col >= M) {
        return;
    }

    // 1) Get CSR range of nonzeros for row in Xtrans
    let startX = row_ptrX[row];
    let endX   = row_ptrX[row + 1u];

    // 2) Get CSC range of nonzeros for column in MM
    let startMM = col_ptrMM[col];
    let endMM   = col_ptrMM[col + 1u];

    var iX  = startX;
    var iMM = startMM;
    var sum = 0.0;

    // 3) Merge intersection on "feature" index
    loop {
        if (iX >= endX || iMM >= endMM) {
            break;
        }
        let fx  = col_idxX[iX];     // feature index from X
        let fmm = row_idxMM[iMM];   // feature index from MM

        if (fx < fmm) {
            iX = iX + 1u;
        } else if (fx > fmm) {
            iMM = iMM + 1u;
        } else {
            // match => multiply
            sum = sum + (valX[iX] * valMM[iMM]);
            iX  = iX + 1u;
            iMM = iMM + 1u;
        }
    }

    // 4) Store final result in SM (S x M)
    SM[(row * M) + col] = sum;
}
