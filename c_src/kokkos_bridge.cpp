// c_src/kokkos_bridge.cpp

#include <Kokkos_Core.hpp>
#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_spgemm.hpp>
#include <cstdio>

//------------------------------------
// Our handle holds a pointer to a Kokkos CrsMatrix on the heap.
struct MyKokkosMatrixHandle {
  using execution_space = Kokkos::DefaultExecutionSpace;
  using memory_space    = execution_space::memory_space;

  using scalar_t  = float; // or double, if you want
  using lno_t     = int;   // local indices
  using size_type = int;   // row offsets

  KokkosSparse::CrsMatrix<scalar_t, lno_t,
      Kokkos::Device<execution_space, memory_space>,
      void, size_type>* crsMat;
};

extern "C" {

//------------------------------------
// 1) Create a Kokkos CrsMatrix from user-provided CSR arrays
void* create_kokkos_crs_matrix(
    int nrows,
    int ncols,
    int nnz,
    const int* row_map,
    const int* col_idx,
    const float* values)
{
    if (!Kokkos::is_initialized()) {
        Kokkos::initialize();
    }

    MyKokkosMatrixHandle* handle = new MyKokkosMatrixHandle();

    using exec_space = MyKokkosMatrixHandle::execution_space;
    using mem_space  = MyKokkosMatrixHandle::memory_space;
    using size_type  = MyKokkosMatrixHandle::size_type;
    using lno_t      = MyKokkosMatrixHandle::lno_t;
    using scalar_t   = MyKokkosMatrixHandle::scalar_t;

    // Allocate device Views
    Kokkos::View<size_type*, mem_space> rowmap_view("rowmap", nrows+1);
    Kokkos::View<lno_t*,     mem_space> colidx_view("colidx", nnz);
    Kokkos::View<scalar_t*,  mem_space> vals_view ("vals",   nnz);

    // Host mirrors
    auto rowmap_host = Kokkos::create_mirror_view(rowmap_view);
    auto colidx_host = Kokkos::create_mirror_view(colidx_view);
    auto vals_host   = Kokkos::create_mirror_view(vals_view);

    // Copy from user arrays
    for(int i=0; i<(nrows+1); i++){
        rowmap_host(i) = row_map[i];
    }
    for(int i=0; i<nnz; i++){
        colidx_host(i) = col_idx[i];
        vals_host(i)   = values[i];
    }

    // Copy to device
    Kokkos::deep_copy(rowmap_view, rowmap_host);
    Kokkos::deep_copy(colidx_view, colidx_host);
    Kokkos::deep_copy(vals_view,   vals_host);

    // Build CrsMatrix
    auto mat = new KokkosSparse::CrsMatrix<scalar_t, lno_t,
       Kokkos::Device<exec_space, mem_space>, void, size_type>(
        "MyMatrix", nrows, ncols, nnz,
        vals_view, rowmap_view, colidx_view
    );

    handle->crsMat = mat;
    return (void*)handle;
}

//------------------------------------
// 2) Destroy a Kokkos CrsMatrix handle
void destroy_kokkos_crs_matrix(void* ptr)
{
    if(!ptr) return;
    auto* handle = reinterpret_cast<MyKokkosMatrixHandle*>(ptr);
    if(handle->crsMat){
        delete handle->crsMat;
        handle->crsMat = nullptr;
    }
    delete handle;
    // Optional: Kokkos::finalize() if no further usage
}

//------------------------------------
// 3) spGEMM: C = A x B (both sparse).
// Returns a new handle to the product matrix.
void* spgemm_kokkos(void* A_ptr, void* B_ptr)
{
    if (!Kokkos::is_initialized()) {
        Kokkos::initialize();
    }

    using handle_t = MyKokkosMatrixHandle;
    using namespace KokkosSparse;

        
    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space    = execution_space::memory_space;

    using crsMat_t = CrsMatrix<
        float, // scalar type
        int,   // local index type
        Kokkos::Device<execution_space, memory_space>,
        void,  // memory trait (default)
        int    // size_type
    >;

    auto A_handle = reinterpret_cast<MyKokkosMatrixHandle*>(A_ptr);
    auto B_handle = reinterpret_cast<MyKokkosMatrixHandle*>(B_ptr);
    if(!A_handle || !B_handle) return nullptr;

    crsMat_t* A = A_handle->crsMat;
    crsMat_t* B = B_handle->crsMat;
    if(!A || !B) return nullptr;

    int m = A->numRows();
    int kA = A->numCols();
    int kB = B->numRows();
    int n  = B->numCols();
    if(kA != kB) {
        // dimension mismatch
        return nullptr;
    }

    KokkosKernels::Experimental::KokkosKernelsHandle<
        int,              // Size type
        int,              // Local index type
        float,            // Scalar type
        execution_space,  // Execution space
        memory_space,     // Temporary memory space
        memory_space      // Persistent memory space
    > kkHandle;

    // Set up SPGEMM operation on the handle
    kkHandle.create_spgemm_handle(KokkosSparse::SPGEMMAlgorithm::SPGEMM_DEFAULT);

    crsMat_t C; // Initialize an empty CrsMatrix for C

    // Symbolic
    spgemm_symbolic(
        kkHandle,
        *A, false,
        *B, false,
        C
    );

    // Get the number of nonzeros in C
    size_t c_nnz = kkHandle.get_spgemm_handle()->get_c_nnz();

    // Resize C's values array to match the computed nonzeros
    C.values = typename crsMat_t::values_type("C_values", c_nnz);


    // Numeric
    spgemm_numeric(
        kkHandle,
        *A, false,
        *B, false,
        C
    );

    auto C_handle = new MyKokkosMatrixHandle();
    C_handle->crsMat = new crsMat_t(C); // Copy C into the handle
    return (void*)C_handle;
}

//------------------------------------
// 4) Export the matrix data (CSR) back to Rust arrays (on host).
// The user must pass in pointers to enough space for row_map (nrows+1),
// col_idx (nnz), and values (nnz).
// We'll fill out nrows, ncols, nnz, plus copy arrays out.
int export_kokkos_crs_matrix(
    void* mat_ptr,
    int* out_nrows,
    int* out_ncols,
    int* out_nnz,
    int* row_map_out,
    int* col_idx_out,
    float* vals_out
)
{
    if(!mat_ptr) return -1;

    auto handle = reinterpret_cast<MyKokkosMatrixHandle*>(mat_ptr);
    if(!handle->crsMat) return -2;

    using crsMat_t = KokkosSparse::CrsMatrix<
        MyKokkosMatrixHandle::scalar_t,
        MyKokkosMatrixHandle::lno_t,
        Kokkos::Device<MyKokkosMatrixHandle::execution_space,
                       MyKokkosMatrixHandle::memory_space>,
        void,
        MyKokkosMatrixHandle::size_type>;

    crsMat_t* M = handle->crsMat;

    int nrows = M->numRows();
    int ncols = M->numCols();
    int nnz   = M->nnz();

    *out_nrows = nrows;
    *out_ncols = ncols;
    *out_nnz   = nnz;

    // Copy device -> host
    // row_map, entries, values are Kokkos::Views
    auto rowmap_dev = M->graph.row_map;
    auto entries_dev= M->graph.entries;
    auto vals_dev   = M->values;

    // create host mirrors
    auto rowmap_host = Kokkos::create_mirror_view(rowmap_dev);
    auto entries_host= Kokkos::create_mirror_view(entries_dev);
    auto vals_host   = Kokkos::create_mirror_view(vals_dev);

    Kokkos::deep_copy(rowmap_host, rowmap_dev);
    Kokkos::deep_copy(entries_host, entries_dev);
    Kokkos::deep_copy(vals_host,   vals_dev);

    // Write them into user buffers
    for(int i=0; i<(nrows+1); i++){
        row_map_out[i] = rowmap_host(i);
    }
    for(int i=0; i<nnz; i++){
        col_idx_out[i] = entries_host(i);
        vals_out[i]    = vals_host(i);
    }

    return 0; // success
}

} // extern "C"
