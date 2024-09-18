#include <iostream>
#include <vector>
#include <tuple>
#include <random>
#include <chrono>
#include <omp.h>
#include <algorithm>
#include <execution> 
#include <cassert>
#include <unordered_set>


// Struktura za CSR matricu
struct CRSMatrix
{
    int rows; // number of rows
    int cols; // number of columns
    
    std::vector<double> values; // non-zero elements
    std::vector<int> colIndex; // column indices
    std::vector<int> rowPtr; // row ptr
};


// Struktura za dense matricu
struct DenseMatrix {
    int rows; 
    int cols; 
    std::vector<std::vector<double>> data; 

    // Constructor
    DenseMatrix(int r, int c) : rows(r), cols(c), data(r, std::vector<double>(c, 0.0)) {}
};

// Funkcija za ispis CSR matrice
void print_csr(const CRSMatrix& matrix) {
    std::cout << "Number of rows: " << matrix.rows << std::endl;
    std::cout << "Number of columns: " << matrix.cols << std::endl;
    std::cout << "Number of non-zero elements: " << matrix.values.size() << std::endl;

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

// Funkcija za generiranje slučajne sparse matrice
CRSMatrix generate_sparse_matrix(int size, double sparsity) {
    CRSMatrix matrix;
    matrix.rows = size;
    matrix.cols = size;
    matrix.rowPtr.resize(size + 1, 0);

    int nnz = static_cast<int>(size * size * sparsity);    
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

CRSMatrix generate_sparse_matrix_equal(int size, double sparsity) {
    CRSMatrix matrix;
    matrix.rows = size;
    matrix.cols = size;
    matrix.rowPtr.resize(size + 1, 0);

    int nnz = static_cast<int>(size * size * sparsity);
    matrix.values.resize(nnz);
    matrix.colIndex.resize(nnz);

    int nnz_per_row = nnz / size; // Broj nenultih elemenata po redu
    int remaining_nnz = nnz % size; // Ostatak nenultih elemenata

    std::default_random_engine generator;
    std::uniform_int_distribution<int> col_dist(0, size - 1);

    int index = 0;

    for (int i = 0; i < size; ++i) {
        int row_nnz = nnz_per_row + (i < remaining_nnz ? 1 : 0); // Dodaj po jedan dodatni element u prvih nekoliko redova

        matrix.rowPtr[i] = index;

        std::unordered_set<int> used_cols; // Skup za praćenje korišćenih kolona kako bi se izbegli duplikati
        for (int j = 0; j < row_nnz; ++j) {
            int col;
            do {
                col = col_dist(generator);
            } while (used_cols.find(col) != used_cols.end()); // Osiguraj da nema duplih kolona u istom redu

            used_cols.insert(col);
            matrix.colIndex[index] = col;
            matrix.values[index] = 1;
            ++index;
        }
    }

    matrix.rowPtr[size] = nnz; // Poslednji element rowPtr pokazuje na kraj

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

DenseMatrix multiply_csr(const CRSMatrix& A, const CRSMatrix& B_transposed) {
    assert(A.cols == B_transposed.rows);

    DenseMatrix C(A.rows, B_transposed.cols);

    for (int i = 0; i < A.rows; ++i) {
        for (int j = A.rowPtr[i]; j < A.rowPtr[i + 1]; ++j) {
            int colA = A.colIndex[j];
            int valA = A.values[j];

            for (int k = B_transposed.rowPtr[colA]; k < B_transposed.rowPtr[colA + 1]; ++k) {
                int colB = B_transposed.colIndex[k];
                int valB = B_transposed.values[k];

                C.data[i][colB] += valA * valB;
            }
        }
    }

    return C;
}

CRSMatrix dense_to_csr(const DenseMatrix& dense) {
    CRSMatrix csr;
    csr.rows = dense.rows;
    csr.cols = dense.cols;

    // Initialize rowPtr with zeros (size: rows + 1)
    csr.rowPtr.resize(csr.rows + 1, 0);

    // Temporary vectors to store column indices and values
    std::vector<int> tempColIndex;
    std::vector<double> tempValues;

    // Populate row pointers and collect column indices and values
    for (int i = 0; i < dense.rows; ++i) {
        int rowStart = tempValues.size(); // Starting index for the current row in values/colIndices

        for (int j = 0; j < dense.cols; ++j) {
            if (dense.data[i][j] != 0) {
                tempColIndex.push_back(j);
                tempValues.push_back(dense.data[i][j]);
            }
        }

        // Update rowPtr for the next row
        csr.rowPtr[i + 1] = tempValues.size();
    }

    // Assign column indices and values
    csr.colIndex = std::move(tempColIndex);
    csr.values = std::move(tempValues);

    return csr;
}


// ovo ne radi kako treba radi ali zna da zamijeni skroz redove 
CRSMatrix dense_to_csr_parallel(const DenseMatrix& dense) {
    CRSMatrix csr;
    csr.rows = dense.rows;
    csr.cols = dense.cols;

    // Initialize rowPtr with zeros (size: rows + 1)
    csr.rowPtr.resize(csr.rows + 1, 0);

    // Prepare temporary vectors for the final result
    std::vector<int> tempColIndex;
    std::vector<double> tempValues;

    // Resize the temporary vectors to accommodate the worst-case scenario
    tempColIndex.reserve(dense.rows * dense.cols);
    tempValues.reserve(dense.rows * dense.cols);

#pragma omp parallel
    {
        // Thread-local vectors to collect results
        std::vector<int> localColIndex;
        std::vector<double> localValues;

#pragma omp for
        for (int i = 0; i < dense.rows; ++i) {
            localColIndex.clear();
            localValues.clear();

            for (int j = 0; j < dense.cols; ++j) {
                if (dense.data[i][j] != 0) {
                    localColIndex.push_back(j);
                    localValues.push_back(dense.data[i][j]);
                }
            }

            // Update thread-local vectors with results from this row
#pragma omp critical
            {
                csr.rowPtr[i + 1] = csr.rowPtr[i] + localValues.size();
                tempColIndex.insert(tempColIndex.end(), localColIndex.begin(), localColIndex.end());
                tempValues.insert(tempValues.end(), localValues.begin(), localValues.end());
            }
        }
    }

    // Finalize rowPtr
    csr.rowPtr[csr.rows] = tempValues.size();

    // Assign column indices and values
    csr.colIndex = std::move(tempColIndex);
    csr.values = std::move(tempValues);

    return csr;
}


DenseMatrix multiply_csr_parallel(const CRSMatrix& A, const CRSMatrix& B_transposed) {
    assert(A.cols == B_transposed.rows);

    DenseMatrix C(A.rows, B_transposed.cols);

#pragma omp parallel for
    for (int i = 0; i < A.rows; ++i) {
        for (int j = A.rowPtr[i]; j < A.rowPtr[i + 1]; ++j) {
            int colA = A.colIndex[j];
            int valA = A.values[j];

            for (int k = B_transposed.rowPtr[colA]; k < B_transposed.rowPtr[colA + 1]; ++k) {
                int colB = B_transposed.colIndex[k];
                int valB = B_transposed.values[k];

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

int testMult() {

    CRSMatrix A;
    A.rows = 3;
    A.cols = 3;    
    A.rowPtr = { 0, 3, 6, 9};
    A.colIndex = { 0, 1, 2, 0, 1, 2, 0, 1, 2};
    A.values = { 1, 2, 3, 4, 5, 6, 7, 8, 9};  
  
    DenseMatrix C = multiply_csr_parallel(A, A);    
        
    std::cout << "Resulting matrix C after multiplying A and transposed B:\n";
    //print_dense(C);

    auto CC = dense_to_csr(C);
    print_csr(CC);

    return 0;
}


int count_non_zero_elements(const DenseMatrix& matrix) {
    int count = 0;
    for (const auto& row : matrix.data) {
        for (const auto& value : row) {
            if (value != 0.0) {
                ++count;
            }
        }
    }
    return count;
}

int main() {

    int size = 10000;
    std::vector<int> sizesVector;
    std::vector<double> time;
    std::vector<int> procentMatrix;
    std::vector<int> elementsA;
    std::vector<int> elementsR;
    for (int n = 1; n <= 10; ++n) {  
        double sparsity = n / 100.0; 
        std::cout << "Test za sparsity: " << sparsity * 100 << "%\n";

        CRSMatrix A = generate_sparse_matrix(size, sparsity);

        int repeat = 5;
        double averageTime = 0.0;
        long number = 0;
        for (int k = 1; k <= repeat; ++k) {
            double timeParallel = measure_time([&]() {
                DenseMatrix C = multiply_csr_parallel(A, A);
                auto CC = dense_to_csr(C);                
            });
            
            averageTime += timeParallel;
        }

        averageTime /= repeat;
        sizesVector.push_back(size);
        procentMatrix.push_back(n);
        elementsA.push_back(A.values.size());
        elementsR.push_back(number);
        time.push_back(averageTime);

        std::cout << "Prosjek vremena za sparsity " << n << "%: " << averageTime << " sekundi\n";
    }
    
    std::cout << "----------------------------------------------------------------" << std::endl;
    std::cout << " Rows | Time | Procent of nnz | Number of values in A matrix | Number of values in result matrix" << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;

    for (size_t i = 0; i < sizesVector.size(); ++i) {
        std::cout << sizesVector[i] << "   | " << time[i] << " s       | " << procentMatrix[i] << "%        | " << elementsA[i] << " | "  << elementsR[i] << std::endl;
    }

    std::cout << "----------------------------------------------------------------" << std::endl;

    return 0;
}

    
