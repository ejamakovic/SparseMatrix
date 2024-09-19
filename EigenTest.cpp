#include <Eigen/Sparse>
#include <iostream>
#include <chrono>
#include <random>

using namespace Eigen;

// Function to measure the time of matrix multiplication
template <typename Func>
double measure_time(Func func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    return duration.count();
}

// Function to generate a sparse matrix with the given size and sparsity
SparseMatrix<double> generate_sparse_matrix(int size, double sparsity) {
    SparseMatrix<double> matrix(size, size);

    std::default_random_engine generator;
    std::uniform_real_distribution<double> value_dist(0.0, 1.0);
    std::uniform_int_distribution<int> index_dist(0, size - 1);

    // Triplet list to hold the non-zero values
    std::vector<Triplet<double>> triplet_list;

    int nnz = static_cast<int>(size * size * sparsity);  // Number of non-zero elements

    for (int i = 0; i < nnz; ++i) {
        int row = index_dist(generator);
        int col = index_dist(generator);
        double value = value_dist(generator);

        triplet_list.push_back(Triplet<double>(row, col, value));
    }

    // Build the sparse matrix from triplet list
    matrix.setFromTriplets(triplet_list.begin(), triplet_list.end());
    return matrix;
}

void benchmark_eigen(int size, double sparsity) {
    // Generate sparse matrices
    SparseMatrix<double> A = generate_sparse_matrix(size, sparsity);
    SparseMatrix<double> B = generate_sparse_matrix(size, sparsity);

    // Time Eigen's sparse matrix multiplication
    double time_eigen = measure_time([&]() {
        SparseMatrix<double> C = A * B;
        });

    std::cout << "Eigen multiplication time: " << time_eigen << " seconds\n";
}

int main() {
    int size = 10000;          // Example matrix size
    double sparsity = 0.01;    // Example sparsity (1% non-zero elements)

    benchmark_eigen(size, sparsity);

    return 0;
}
