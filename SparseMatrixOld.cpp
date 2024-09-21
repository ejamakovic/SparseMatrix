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
#include <iomanip>


// Struktura za CSR matricu
struct CRSMatrix
{
    int rows; 
    int cols; 
    std::vector<double> values; 
    std::vector<int> colIndex; 
    std::vector<int> rowPtr; 
};


// Struktura za dense matricu
struct DenseMatrix {
    int rows;
    int cols;
    std::vector<std::vector<double>> data;

    DenseMatrix() {}   
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

CRSMatrix generate_sparse_matrix(int size, double sparsity) {
    CRSMatrix matrix;
    matrix.rows = size;
    matrix.cols = size;
    matrix.rowPtr.resize(size + 1, 0);

    int nnz = static_cast<int>(size * size * sparsity);
    matrix.values.resize(nnz);
    matrix.colIndex.resize(nnz);

    std::vector<int> row_lengths(size, 0);
    std::vector<std::vector<int>> row_columns(size);
    std::default_random_engine generator;
    std::uniform_int_distribution<int> row_dist(0, size - 1);

    for (int i = 0; i < nnz; ++i) {
        int row = row_dist(generator);
        int col = row_dist(generator);
        row_columns[row].push_back(col); 
        ++row_lengths[row];
    }
    
    for (int row = 0; row < size; ++row) {
        std::sort(row_columns[row].begin(), row_columns[row].end());
    }

    int index = 0;
    for (int i = 0; i < size; ++i) {
        matrix.rowPtr[i] = index;
        for (int col : row_columns[i]) {
            matrix.colIndex[index] = col;
            matrix.values[index] = 1;
            ++index;
        }
    }

    matrix.rowPtr[size] = nnz; 

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
        std::vector<int> cols; // Privremeni vektor za čuvanje kolona

        for (int j = 0; j < row_nnz; ++j) {
            int col;
            do {
                col = col_dist(generator);
            } while (used_cols.find(col) != used_cols.end()); // Osiguraj da nema duplih kolona u istom redu

            used_cols.insert(col);
            cols.push_back(col); // Sačuvaj kolonu u privremeni vektor
        }

        // Sortiraj kolone
        std::sort(cols.begin(), cols.end());

        // Dodaj sortirane kolone i vrednosti u CSR matricu
        for (int col : cols) {
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


CRSMatrix dense_to_csr(const DenseMatrix& dense) {
    CRSMatrix csr;
    csr.rows = dense.rows;
    csr.cols = dense.cols;    
    csr.rowPtr.resize(csr.rows + 1, 0);
   
    std::vector<int> tempColIndex;
    std::vector<double> tempValues;

    
    for (int i = 0; i < dense.rows; ++i) {
        int rowStart = tempValues.size(); 

        for (int j = 0; j < dense.cols; ++j) {
            if (dense.data[i][j] != 0) {
                tempColIndex.push_back(j);
                tempValues.push_back(dense.data[i][j]);
            }
        }
        
        csr.rowPtr[i + 1] = tempValues.size();
    }
    
    csr.colIndex = std::move(tempColIndex);
    csr.values = std::move(tempValues);

    return csr;
}

CRSMatrix dense_to_csr_parallel(const DenseMatrix& dense) {
    CRSMatrix csr;
    csr.rows = dense.rows;
    csr.cols = dense.cols; 
    csr.rowPtr.resize(csr.rows + 1, 0);

    // First pass: Calculate the number of non-zero elements per row in parallel
#pragma omp parallel for
    for (int i = 0; i < dense.rows; ++i) {
        for (int j = 0; j < dense.cols; ++j) {
            if (dense.data[i][j] != 0) {
#pragma omp atomic
                csr.rowPtr[i + 1]++;
            }
        }
    }

    // Prefix sum (cumulative sum) to get rowPtr positions
    for (int i = 1; i <= dense.rows; ++i) {
        csr.rowPtr[i] += csr.rowPtr[i - 1];
    }

    // Allocate space for colIndex and values based on the total number of non-zero elements
    int totalNonZeros = csr.rowPtr[dense.rows];
    csr.colIndex.resize(totalNonZeros);
    csr.values.resize(totalNonZeros);

    // Temporary array to track the current write position for each row
    std::vector<int> currentPos(csr.rows, 0);

    // Initialize currentPos to the starting positions of each row
#pragma omp parallel for
    for (int i = 0; i < csr.rows; ++i) {
        currentPos[i] = csr.rowPtr[i];
    }

    // Second pass: Populate colIndex and values in parallel, writing to the correct positions
#pragma omp parallel for
    for (int i = 0; i < dense.rows; ++i) {
        for (int j = 0; j < dense.cols; ++j) {
            if (dense.data[i][j] != 0) {
                int pos = currentPos[i]; // Get the current position for this row

                csr.colIndex[pos] = j;
                csr.values[pos] = dense.data[i][j];

                // Increment the current position for the next non-zero element in this row
                currentPos[i]++;
            }
        }
    }

    return csr;
}


DenseMatrix multiply_csr(const CRSMatrix& A, const CRSMatrix& B) {
    assert(A.cols == B.rows);

    DenseMatrix C(A.rows, B.cols);

    for (int i = 0; i < A.rows; ++i) {
        for (int j = A.rowPtr[i]; j < A.rowPtr[i + 1]; ++j) {
            int colA = A.colIndex[j];
            double valA = A.values[j];
            int nextRow = B.rowPtr[colA + 1];
            for (int k = B.rowPtr[colA]; k < nextRow; ++k) {
                int colB = B.colIndex[k];
                double valB = B.values[k];

                C.data[i][colB] += valA * valB;
            }
        }
    }

    return C;
}

DenseMatrix multiply_csr_parallel(const CRSMatrix& A, const CRSMatrix& B) {
    assert(A.cols == B.rows);

    DenseMatrix C(A.rows, B.cols);

#pragma omp parallel for
    for (int i = 0; i < A.rows; ++i) {
        for (int j = A.rowPtr[i]; j < A.rowPtr[i + 1]; ++j) {
            int colA = A.colIndex[j];
            double valA = A.values[j];
            int nextRow = B.rowPtr[colA + 1];
            for (int k = B.rowPtr[colA]; k < nextRow; ++k) {
                int colB = B.colIndex[k];
                double valB = B.values[k];

                C.data[i][colB] += valA * valB;
            }
        }
    }

    return C;
}


int testMult() {
    CRSMatrix A;
    A.rows = 3;
    A.cols = 3;
    A.rowPtr = { 0, 3, 6, 9 };
    A.colIndex = { 0, 1, 2, 0, 1, 2, 0, 1, 2 };
    A.values = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };

    DenseMatrix C = multiply_csr_parallel(A, A);
    auto CC = dense_to_csr_parallel(C);
    print_csr(CC);

    return 0;

    //Tacan rezultat mnozenja:
    // 
    //Row Pointer (rowPtr): 0 3 6 9
    //Column Indices(colIndex) : 0 1 2 0 1 2 0 1 2
    //Values(val) : 30 36 42 66 81 96 102 126 150
}

int main() {
    testMult();


    int size = 10000;
    std::vector<int> sizesVector;
    std::vector<double> time;
    std::vector<int> procentMatrix;
    std::vector<int> elementsR;
    for (int n = 1; n <= 10; ++n) {
        double sparsity = n / 100.0;
        std::cout << "Test za sparsity: " << sparsity * 100 << "%\n";

        int repeat = 1;
        double averageTime = 0.0;
        CRSMatrix CC;
        for (int k = 1; k <= repeat; ++k) {
            CRSMatrix A = generate_sparse_matrix_equal(size, sparsity);
            //print_csr(A);
            CRSMatrix B = generate_sparse_matrix(size, sparsity);
            double timeParallel = measure_time([&]() {
                DenseMatrix C = multiply_csr_parallel(A, B);
                CC = dense_to_csr_parallel(C);
                });

            averageTime += timeParallel;
        }

        averageTime /= repeat;
        sizesVector.push_back(size);
        procentMatrix.push_back(n);
        elementsR.push_back(CC.values.size());
        time.push_back(averageTime);

        std::cout << "Prosjek vremena za rijetkost " << n << "%: " << averageTime << " sekundi\n";
    }

    std::cout << "-------------------------------------------------------------------------------------------" << std::endl;
    std::cout << " Size of A |    Time    | Procent of nnz in A matrix | Number of values in result matrix " << std::endl;
    std::cout << "-------------------------------------------------------------------------------------------" << std::endl;

    for (size_t i = 0; i < sizesVector.size(); ++i) {
        std::cout << "   " << sizesVector[i] << "   |  " << std::fixed << std::setprecision(4) << time[i] << " s  |             " << procentMatrix[i] << "%             |              ";
        std::cout << elementsR[i];
        std::cout << std::endl;
    }

    std::cout << "-------------------------------------------------------------------------------------------" << std::endl;

    return 0;
}
