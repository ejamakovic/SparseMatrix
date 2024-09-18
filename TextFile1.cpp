#include <iostream>
#include <vector>
#include <tuple>
#include <random>
#include <chrono>
#include <omp.h>
#include <algorithm>
#include <execution> 
#include <cassert>

// Struktura za CSR matricu
struct CRSMatrix
{
    int rows; // number of rows
    int cols; // number of columns
    int nnz; // number of non-zero elements
    std::vector<double> values; // non-zero elements
    std::vector<int> colIndex; // column indices
    std::vector<int> rowPtr; // row ptr
};


// Struktura za dense matricu
struct DenseMatrix {
    int rows; // number of rows
    int cols; // number of columns
    std::vector<std::vector<double>> data; // matrix data

    // Constructor
    DenseMatrix(int r, int c) : rows(r), cols(c), data(r, std::vector<double>(c, 0.0)) {}
};

// Funkcija za ispis CSR matrice
void print_csr(const CRSMatrix& matrix) {
    std::cout << "Number of rows: " << matrix.rows << std::endl;
    std::cout << "Number of columns: " << matrix.cols << std::endl;
    std::cout << "Number of non-zero elements: " << matrix.nnz << std::endl;

    std::cout << "\nRow Pointer (rowPtr): ";
    for (const auto& val : matrix.rowPtr) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    std::cout << "Column Indices (colIndex): ";
    for (const auto& val : matrix.colIndex) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    std::cout << "Values (val): ";
    for (const auto& val : matrix.values) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}

// Funkcija za paralelno transponiranje CSR matrice
template <typename SIZE, typename R, typename C, typename V>
auto ParallelTranspose(const SIZE rows, const SIZE cols, const SIZE nnz,
    const SIZE base, const R& ai, const C& aj, const V& av) {
    using ROWTYPE = typename std::decay<decltype(ai[0])>::type;
    using COLTYPE = typename std::decay<decltype(aj[0])>::type;
    using VALTYPE = typename std::decay<decltype(av[0])>::type;
    const SIZE cols_transpose = rows;
    const SIZE rows_transpose = cols;

    std::vector<ROWTYPE> ai_transpose(rows_transpose + 1);
    std::vector<COLTYPE> aj_transpose(nnz);
    std::vector<VALTYPE> av_transpose(nnz);

    ai_transpose[0] = base;

    std::vector<std::vector<ROWTYPE>> threadPrefixSum(omp_get_max_threads(), std::vector<ROWTYPE>(rows_transpose));

#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const int nthreads = omp_get_num_threads();

        std::vector<ROWTYPE>& threadSum = threadPrefixSum[tid];

#pragma omp for
        for (SIZE i = 0; i < rows; i++) {
            for (ROWTYPE j = ai[i] - base; j < ai[i + 1] - base; j++) {
                threadSum[aj[j] - base]++;
            }
        }

#pragma omp barrier
#pragma omp for
        for (SIZE rowID = 0; rowID < rows_transpose; rowID++) {
            ai_transpose[rowID + 1] = 0;
            for (int t = 0; t < nthreads; t++) {
                ai_transpose[rowID + 1] += threadPrefixSum[t][rowID];
            }
        }

#pragma omp barrier
#pragma omp master
        {
            std::inclusive_scan(std::execution::par, ai_transpose.begin(), ai_transpose.end(), ai_transpose.begin());
        }

#pragma omp barrier
#pragma omp for
        for (SIZE rowID = 0; rowID < rows_transpose; rowID++) {
            ROWTYPE tmp = threadPrefixSum[0][rowID];
            threadPrefixSum[0][rowID] = ai_transpose[rowID];
            for (int t = 1; t < nthreads; t++) {
                std::swap(threadPrefixSum[t][rowID], tmp);
                threadPrefixSum[t][rowID] += threadPrefixSum[t - 1][rowID];
            }
        }

#pragma omp barrier

#pragma omp for
        for (SIZE i = 0; i < rows; i++) {
            for (ROWTYPE j = ai[i] - base; j < ai[i + 1] - base; j++) {
                const COLTYPE idx = threadPrefixSum[tid][aj[j] - base]++ - base;
                aj_transpose[idx] = i + base;
                av_transpose[idx] = av[j];
            }
        }
    }

    return std::make_tuple(std::move(ai_transpose), std::move(aj_transpose), std::move(av_transpose));
}

// Funkcija za generiranje sluèajne sparse matrice
CRSMatrix generate_sparse_matrix(int size, double sparsity) {
    CRSMatrix matrix;
    matrix.rows = size;
    matrix.cols = size;
    matrix.rowPtr.resize(size + 1, 0);

    int nnz = static_cast<int>(size * size * sparsity);
    matrix.nnz = nnz;
    matrix.values.resize(nnz);
    matrix.colIndex.resize(nnz);

    std::vector<int> row_lengths(size, 0);
    std::default_random_engine generator;
    std::uniform_int_distribution<int> row_dist(0, size - 1);

    for (int i = 0; i < nnz; ++i) {
        int row = row_dist(generator);
        int col = row_dist(generator);
        matrix.colIndex[i] = col;
        matrix.values[i] = 1;
        ++row_lengths[row];
    }

    for (int i = 1; i <= size; ++i) {
        matrix.rowPtr[i] = matrix.rowPtr[i - 1] + row_lengths[i - 1];
    }

    return matrix;
}

// Funkcija za mjerenje vremena
template <typename Func>
auto measure_time(Func func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    return duration.count();
}

int test() {
    // Define the example matrix in CSR format
    std::vector<int> values{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    std::vector<int> cols{ 0, 1, 2, 0, 1, 2, 0, 1, 2 };
    std::vector<int> rowP{ 0, 3, 6, 9 }; // rowPtr

    // Number of rows and columns
    int rows = 3;
    int cols_count = 3;
    int nnz = values.size();
    int base = 0; // base for CSR format

    // Call the ParallelTranspose function
    auto [ai_transpose, aj_transpose, av_transpose] = ParallelTranspose(rows, cols_count, nnz, base, rowP, cols, values);

    // Print the transposed matrix in CSR format
    std::cout << "Transposed matrix in CSR format:" << std::endl;
    std::cout << "rowPtr: ";
    for (const auto& val : ai_transpose) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    std::cout << "colIndex: ";
    for (const auto& val : aj_transpose) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    std::cout << "values: ";
    for (const auto& val : av_transpose) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}

// Testna funkcija
void test_generate_sparse_matrix() {
    int size = 3; // Dimenzije matrice
    double sparsity = 0.5; // Sparsity faktor (40% ne-nultih elemenata)

    CRSMatrix matrix = generate_sparse_matrix(size, sparsity);

    // Proveri dimenzije
    assert(matrix.rows == size);
    assert(matrix.cols == size);

    // Proveri broj ne-nultih elemenata
    int expected_nnz = static_cast<int>(size * size * sparsity);
    assert(matrix.nnz == expected_nnz);

    // Proveri da li su rowPtr, colIndex i val ispravni
    for (int i = 0; i < size; ++i) {
        assert(matrix.rowPtr[i + 1] >= matrix.rowPtr[i]); // rowPtr mora biti monotono rastuæa
    }

    // Proveri da li su svi koloni unutar opsega
    for (int i = 0; i < matrix.nnz; ++i) {
        assert(matrix.colIndex[i] >= 0 && matrix.colIndex[i] < size); // Kolone moraju biti unutar opsega
        assert(matrix.values[i] >= 0.1 && matrix.values[i] <= 1.0); // Vrednosti moraju biti unutar opsega
    }

    std::cout << "Test passed: Sparse matrix generated successfully." << std::endl;
    print_csr(matrix);
}

// Funkcija za množenje CSR matrice sa njenom transponovanom verzijom
DenseMatrix multiply_csr_with_transpose(const CRSMatrix& A, const CRSMatrix& B_transposed) {
    assert(A.cols == B_transposed.rows); // Ensure matrix dimensions are compatible for multiplication

    DenseMatrix C(A.rows, B_transposed.cols);

    // Iterate through each row of A
    for (int i = 0; i < A.rows; ++i) {
        // Iterate through each non-zero element in row i of A
        for (int j = A.rowPtr[i]; j < A.rowPtr[i + 1]; ++j) {
            int colA = A.colIndex[j];
            int valA = A.values[j];

            // Iterate through each non-zero element in column colA of B_transposed
            for (int k = B_transposed.rowPtr[colA]; k < B_transposed.rowPtr[colA + 1]; ++k) {
                int colB = B_transposed.colIndex[k];
                int valB = B_transposed.values[k];

                // Accumulate the product in the result matrix C
                C.data[i][colB] += valA * valB;
            }
        }
    }

    return C;
}

// Funkcija za ispis dense matrice
void print_dense(const DenseMatrix& matrix) {
    std::cout << "Dense Matrix (" << matrix.rows << "x" << matrix.cols << "):\n";
    for (const auto& row : matrix.data) {
        for (const auto& value : row) {
            std::cout << value << " ";
        }
        std::cout << "\n";
    }
}

DenseMatrix transpose_dense(const DenseMatrix& matrix) {
    DenseMatrix transposed(matrix.cols, matrix.rows);
    for (int i = 0; i < matrix.rows; ++i) {
        for (int j = 0; j < matrix.cols; ++j) {
            transposed.data[j][i] = matrix.data[i][j];
        }
    }
    return transposed;
}

int main() {
    // Define matrix A in CSR format
    CRSMatrix A;
    A.rows = 2; A.cols = 2;
    A.rowPtr = { 0, 1, 2 };
    A.colIndex = { 0, 1 };
    A.values = { 1, 1 };
    A.nnz = A.values.size();

    // Define matrix B in CSR format
    CRSMatrix B;
    B.rows = 2; B.cols = 2;
    B.rowPtr = { 0, 0, 2 };
    B.colIndex = { 0, 1 };
    B.values = { 1, 1 };
    B.nnz = B.values.size();

    // Transpose matrix B
    auto [B_transpose_rowPtr, B_transpose_colIndex, B_transpose_val] = ParallelTranspose(B.rows, B.cols, B.nnz, 0, B.rowPtr, B.colIndex, B.values);

    CRSMatrix B_transposed;
    B_transposed.rows = B.rows;
    B_transposed.cols = B.cols;
    B_transposed.rowPtr = std::move(B_transpose_rowPtr);
    B_transposed.colIndex = std::move(B_transpose_colIndex);
    B_transposed.values = std::move(B_transpose_val);

    // Multiply A and transposed B
    DenseMatrix C = multiply_csr_with_transpose(A, B_transposed);

    C = transpose_dense(C);
    print_dense(C);

    return 0;
}

