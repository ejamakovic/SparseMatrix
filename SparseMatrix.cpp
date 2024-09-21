#include <iostream>
#include <tuple>
#include <random>
#include <chrono>
#include <omp.h>
#include <algorithm>
#include <unordered_set>
#include <iomanip>
#include <cassert>

class CRSMatrix;

// Class for Dense Matrix
class DenseMatrix
{
public:
    int rows;
    int cols;
    double **data;

    DenseMatrix(int r = 0, int c = 0) : rows(r), cols(c)
    {
        data = new double *[r];
        for (int i = 0; i < r; ++i)
        {
            data[i] = new double[c]();
        }
    }

    ~DenseMatrix()
    {
        for (int i = 0; i < rows; ++i)
        {
            delete[] data[i];
        }
        delete[] data;
    }

    CRSMatrix to_csr() const;
    CRSMatrix to_csr_parallel() const;
};

// Class for Compressed Row Storage (CRS) Matrix
class CRSMatrix
{
public:
    int rows;
    int cols;
    double *values;
    int *colIndex;
    int *rowPtr;
    int nnz;

    CRSMatrix(int r = 0, int c = 0) : rows(r), cols(c), nnz(0), values(nullptr), colIndex(nullptr)
    {
        rowPtr = new int[r + 1]();
    }

    ~CRSMatrix()
    {
        if (rowPtr)
        {
            delete[] rowPtr;
            rowPtr = nullptr;
        }
        if (values)
        {
            delete[] values;
            values = nullptr;
        }
        if (colIndex)
        {
            delete[] colIndex;
            colIndex = nullptr;
        }
    }

    void allocate(int nonZeros)
    {
        if (values)
        {
            delete[] values;
            values = nullptr;
        }
        if (colIndex)
        {
            delete[] colIndex;
            colIndex = nullptr;
        }

        nnz = nonZeros;
        if (nnz > 0)
        {
            values = new double[nnz];
            colIndex = new int[nnz];
        }
    }

    void print() const
    {
        std::cout << "Number of rows: " << rows << "\n";
        std::cout << "Number of columns: " << cols << "\n";
        std::cout << "Number of non-zero elements: " << nnz << "\n";

        std::cout << "\nRow Pointer (rowPtr): ";
        for (int i = 0; i <= rows; ++i)
            std::cout << rowPtr[i] << " ";
        std::cout << "\nColumn Indices (colIndex): ";
        for (int i = 0; i < nnz; ++i)
            std::cout << colIndex[i] << " ";
        std::cout << "\nValues (values): ";
        for (int i = 0; i < nnz; ++i)
            std::cout << values[i] << " ";
        std::cout << "\n";
    }

    // Function to add an element to the matrix
    void add_element(int row, int col, double value)
    {
        assert(row >= 0 && row < rows && col >= 0 && col < cols && "Invalid indices!");

        int start = rowPtr[row];
        int end = rowPtr[row + 1];

        for (int i = start; i < end; ++i)
        {
            if (colIndex[i] == col)
            {
                values[i] = value;
                return;
            }
        }

        int insert_pos = end;
        for (int i = start; i < end; ++i)
        {
            if (colIndex[i] > col)
            {
                insert_pos = i;
                break;
            }
        }

        double *newValues = new double[nnz + 1];
        int *newColIndex = new int[nnz + 1];

        for (int i = 0; i < insert_pos; ++i)
        {
            newValues[i] = values[i];
            newColIndex[i] = colIndex[i];
        }

        newValues[insert_pos] = value;
        newColIndex[insert_pos] = col;

        for (int i = insert_pos; i < nnz; ++i)
        {
            newValues[i + 1] = values[i];
            newColIndex[i + 1] = colIndex[i];
        }

        delete[] values;
        delete[] colIndex;
        values = newValues;
        colIndex = newColIndex;
        nnz++;

        for (int i = row + 1; i <= rows; ++i)
        {
            rowPtr[i]++;
        }
    }

    // Static method to generate a sparse matrix with equal distribution
    static CRSMatrix generate_equal(int size, double sparsity)
    {
        CRSMatrix matrix(size, size);
        int nnz = static_cast<int>(size * size * sparsity);
        matrix.allocate(nnz);

        int nnz_per_row = nnz / size;
        int remaining_nnz = nnz % size;

        std::default_random_engine generator;
        std::uniform_int_distribution<int> col_dist(0, size - 1);

        int index = 0;

        for (int i = 0; i < size; ++i)
        {
            int row_nnz = nnz_per_row + (i < remaining_nnz ? 1 : 0);
            matrix.rowPtr[i] = index;
            std::unordered_set<int> used_cols;
            std::vector<int> cols;

            for (int j = 0; j < row_nnz; ++j)
            {
                int col;
                do
                {
                    col = col_dist(generator);
                } while (used_cols.find(col) != used_cols.end());
                used_cols.insert(col);
                cols.push_back(col);
            }

            std::sort(cols.begin(), cols.end());
            for (int col : cols)
            {
                matrix.colIndex[index] = col;
                matrix.values[index] = 1.0;
                ++index;
            }
        }
        matrix.rowPtr[size] = nnz;
        return matrix;
    }

    // Static method to generate a random sparse matrix
    static CRSMatrix generate_random(int size, double sparsity)
    {
        CRSMatrix matrix(size, size);
        int nnz = static_cast<int>(size * size * sparsity);
        matrix.allocate(nnz);

        std::default_random_engine generator;
        std::uniform_int_distribution<int> dist(0, size - 1);
        std::vector<std::vector<int>> row_columns(size);

        for (int i = 0; i < nnz; ++i)
        {
            int row = dist(generator);
            int col = dist(generator);
            row_columns[row].push_back(col);
        }

        int index = 0;
        for (int i = 0; i < size; ++i)
        {
            matrix.rowPtr[i] = index;
            std::sort(row_columns[i].begin(), row_columns[i].end());
            for (int col : row_columns[i])
            {
                matrix.colIndex[index] = col;
                matrix.values[index] = 1.0;
                ++index;
            }
        }
        matrix.rowPtr[size] = nnz;
        return matrix;
    }

    // Sequential matrix multiplication
    DenseMatrix multiply(const CRSMatrix &B) const
    {
        assert(cols == B.rows);
        DenseMatrix C(rows, B.cols);

        for (int i = 0; i < rows; ++i)
        {
            for (int j = rowPtr[i]; j < rowPtr[i + 1]; ++j)
            {
                int colA = colIndex[j];
                double valA = values[j];
                int nextRow = B.rowPtr[colA + 1];

                for (int k = B.rowPtr[colA]; k < nextRow; ++k)
                {
                    int colB = B.colIndex[k];
                    double valB = B.values[k];
                    C.data[i][colB] += valA * valB;
                }
            }
        }
        return C;
    }

    // Parallel matrix multiplication using OpenMP
    DenseMatrix multiply_parallel(const CRSMatrix &B) const
    {
        assert(cols == B.rows);
        DenseMatrix C(rows, B.cols);

#pragma omp parallel for
        for (int i = 0; i < rows; ++i)
        {
            for (int j = rowPtr[i]; j < rowPtr[i + 1]; ++j)
            {
                int colA = colIndex[j];
                double valA = values[j];
                int nextRow = B.rowPtr[colA + 1];

                for (int k = B.rowPtr[colA]; k < nextRow; ++k)
                {
                    int colB = B.colIndex[k];
                    double valB = B.values[k];
#pragma omp atomic
                    C.data[i][colB] += valA * valB;
                }
            }
        }
        return C;
    }

    CRSMatrix(const CRSMatrix &other) : rows(other.rows), cols(other.cols), nnz(other.nnz)
    {
        rowPtr = new int[rows + 1];
        std::copy(other.rowPtr, other.rowPtr + rows + 1, rowPtr);

        if (nnz > 0)
        {
            values = new double[nnz];
            colIndex = new int[nnz];
            std::copy(other.values, other.values + nnz, values);
            std::copy(other.colIndex, other.colIndex + nnz, colIndex);
        }
        else
        {
            values = nullptr;
            colIndex = nullptr;
        }
    }

    CRSMatrix &operator=(const CRSMatrix &other)
    {
        if (this != &other)
        {
            rows = other.rows;
            cols = other.cols;
            nnz = other.nnz;

            delete[] rowPtr;
            delete[] values;
            delete[] colIndex;

            rowPtr = new int[rows + 1];
            std::copy(other.rowPtr, other.rowPtr + rows + 1, rowPtr);

            if (nnz > 0)
            {
                values = new double[nnz];
                colIndex = new int[nnz];
                std::copy(other.values, other.values + nnz, values);
                std::copy(other.colIndex, other.colIndex + nnz, colIndex);
            }
            else
            {
                values = nullptr;
                colIndex = nullptr;
            }
        }
        return *this;
    }

    // Move constructor
    CRSMatrix(CRSMatrix &&other) noexcept : rows(other.rows), cols(other.cols), nnz(other.nnz),
                                            values(other.values), colIndex(other.colIndex), rowPtr(other.rowPtr)
    {
        other.values = nullptr;
        other.colIndex = nullptr;
        other.rowPtr = nullptr;
    }

    // Move assignment operator
    CRSMatrix &operator=(CRSMatrix &&other) noexcept
    {
        if (this != &other)
        {
            delete[] values;
            delete[] colIndex;
            delete[] rowPtr;

            rows = other.rows;
            cols = other.cols;
            nnz = other.nnz;
            values = other.values;
            colIndex = other.colIndex;
            rowPtr = other.rowPtr;

            other.values = nullptr;
            other.colIndex = nullptr;
            other.rowPtr = nullptr;
        }
        return *this;
    }
};

// Definition of the DenseMatrix `to_csr` function
CRSMatrix DenseMatrix::to_csr() const
{
    CRSMatrix csr(rows, cols);
    csr.allocate(0); // Initialize with 0 non-zeros

    // First pass: Count non-zero entries in each row
    int nnz = 0;
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            if (data[i][j] != 0)
                nnz++;
        }
    }
    csr.allocate(nnz);

    // Second pass: Populate values and indices
    nnz = 0;
    for (int i = 0; i < rows; ++i)
    {
        csr.rowPtr[i] = nnz;
        for (int j = 0; j < cols; ++j)
        {
            if (data[i][j] != 0)
            {
                csr.values[nnz] = data[i][j];
                csr.colIndex[nnz] = j;
                nnz++;
            }
        }
    }
    csr.rowPtr[rows] = nnz;
    return csr;
}

CRSMatrix DenseMatrix::to_csr_parallel() const
{
    CRSMatrix csr(rows, cols);

    // First pass: Count the number of non-zero elements in parallel
    csr.rowPtr = new int[rows + 1](); // Initialize row pointers to zero

#pragma omp parallel for
    for (int i = 0; i < rows; ++i)
    {
        int nonZeroCount = 0;
        for (int j = 0; j < cols; ++j)
        {
            if (data[i][j] != 0)
            {
                ++nonZeroCount;
            }
        }
        csr.rowPtr[i + 1] = nonZeroCount;
    }

    // Prefix sum to calculate final rowPtr values
    for (int i = 1; i <= rows; ++i)
    {
        csr.rowPtr[i] += csr.rowPtr[i - 1];
    }

    // Allocate memory for non-zero values and column indices
    int nnz = csr.rowPtr[rows];
    csr.allocate(nnz);

    // Second pass: Populate values and column indices
#pragma omp parallel for
    for (int i = 0; i < rows; ++i)
    {
        int index = csr.rowPtr[i];
        for (int j = 0; j < cols; ++j)
        {
            if (data[i][j] != 0)
            {
                csr.values[index] = data[i][j];
                csr.colIndex[index] = j;
                ++index;
            }
        }
    }

    return csr;
}

// Utility to measure execution time
template <typename Func>
double measure_time(Func func)
{
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    return duration.count();
}

int test_multiplication()
{
    CRSMatrix A;
    A.rows = 3;
    A.cols = 3;
    A.rowPtr = new int[4]{0, 3, 6, 9};
    A.colIndex = new int[9]{0, 1, 2, 0, 1, 2, 0, 1, 2};
    A.values = new double[9]{1, 2, 3, 4, 5, 6, 7, 8, 9};

    DenseMatrix C = A.multiply_parallel(A);
    auto CC = C.to_csr_parallel();
    CC.print();
    return 0;

    // Tacan rezultat mnozenja:
    //
    // Row Pointer (rowPtr): 0 3 6 9
    // Column Indices(colIndex) : 0 1 2 0 1 2 0 1 2
    // Values(val) : 30 36 42 66 81 96 102 126 150
}

void preformance_test_without_conversion()
{
    int size = 10000;
    std::vector<int> sizesVector;
    std::vector<double> time;
    std::vector<int> procentMatrix;
    std::vector<int> elementsR;
    std::cout << "Running parallel performance test with matrix size 10000...\n";

    for (int n = 1; n <= 10; ++n)
    {
        double sparsity = n / 100.0;
        std::cout << "Sparsity test: " << sparsity * 100 << "%\n";

        int repeat = 1;
        double averageTime = 0.0;
        CRSMatrix CC;

        for (int k = 1; k <= repeat; ++k)
        {
            CRSMatrix A = CRSMatrix::generate_equal(size, sparsity);

            CRSMatrix B = CRSMatrix::generate_random(size, sparsity);

            double timeParallel = measure_time([&]()
                                               {
                DenseMatrix C = A.multiply_parallel(B);
                CC = C.to_csr_parallel(); });

            averageTime += timeParallel;
        }

        averageTime /= repeat;
        sizesVector.push_back(size);
        procentMatrix.push_back(n);
        elementsR.push_back(CC.nnz);
        time.push_back(averageTime);

        std::cout << "Average time with procent nnz of " << n << "%: " << averageTime << " seconds\n";
    }

    std::cout << "-------------------------------------------------------------------------------------------" << std::endl;
    std::cout << " Size of A |    Time    | Procent of nnz in A matrix | Number of values in result matrix " << std::endl;
    std::cout << "-------------------------------------------------------------------------------------------" << std::endl;

    for (size_t i = 0; i < sizesVector.size(); ++i)
    {
        std::cout << "   " << sizesVector[i] << "   |  " << std::fixed << std::setprecision(4) << time[i] << " s  |             " << procentMatrix[i] << "%             |              ";
        std::cout << elementsR[i];
        std::cout << std::endl;
    }

    std::cout << "-------------------------------------------------------------------------------------------" << std::endl;
}

void preformance_test_with_conversion()
{

    // Tests with larger matrices and different sparsity values
    int size = 10000; // Matrix size for larger tests
    std::cout << "-----------------------------------------" << std::endl;
    std::cout << "Starting performance tests with large matrices (size: " << size << ")..." << std::endl;
    std::cout << "-----------------------------------------" << std::endl;

    std::cout << std::setw(10) << "Sparsity" << " | "
              << std::setw(15) << "Time (Parallel)" << " | "
              << std::setw(15) << "Non-Zero Elements (A)" << " | "
              << std::setw(15) << "Non-Zero Elements (B)" << std::endl;
    std::cout << "-------------------------------------------------------------" << std::endl;

    for (int n = 1; n <= 10; ++n)
    {
        double sparsity = n / 100.0;

        // Generate matrices A and B
        std::cout << "Generating matrix A with sparsity: " << n << "%" << std::endl;
        CRSMatrix A = CRSMatrix::generate_equal(size, sparsity);
        std::cout << "Matrix A generated with " << A.nnz << " non-zero elements." << std::endl;

        std::cout << "Generating matrix B with sparsity: " << n << "%" << std::endl;
        CRSMatrix B = CRSMatrix::generate_random(size, sparsity);
        std::cout << "Matrix B generated with " << B.nnz << " non-zero elements." << std::endl;

        // Measure time for parallel matrix multiplication
        double timeParallel = measure_time([&]()
                                           { DenseMatrix C = A.multiply_parallel(B); });

        std::cout << std::setw(10) << n << "%" << " | "
                  << std::setw(15) << timeParallel << " seconds | "
                  << std::setw(15) << A.nnz << " | "
                  << std::setw(15) << B.nnz << std::endl;
    }

    std::cout << "-----------------------------------------" << std::endl;
    std::cout << "Performance tests completed." << std::endl;
}

// Performance test using sequential multiplication
void preformance_test_sequential_without_conversion()
{
    std::cout << "Running sequential performance test with matrix size 10000...\n";
    int size = 10000;
    std::vector<int> sizesVector;
    std::vector<double> time;
    std::vector<int> procentMatrix;
    std::vector<int> elementsR;

    for (int n = 1; n <= 10; ++n)
    {
        double sparsity = n / 100.0;
        std::cout << "Sparsity test: " << sparsity * 100 << "%\n";

        int repeat = 1;
        double averageTime = 0.0;
        CRSMatrix CC;

        for (int k = 1; k <= repeat; ++k)
        {
            CRSMatrix A = CRSMatrix::generate_equal(size, sparsity);
            CRSMatrix B = CRSMatrix::generate_random(size, sparsity);

            double timeSequential = measure_time([&]()
                                                 {
                DenseMatrix C = A.multiply(B);  
                CC = C.to_csr(); });

            averageTime += timeSequential;
        }

        averageTime /= repeat;
        sizesVector.push_back(size);
        procentMatrix.push_back(n);
        elementsR.push_back(CC.nnz);
        time.push_back(averageTime);

        std::cout << "Average time with procent nnz of " << n << "%: " << averageTime << " seconds\n";
    }

    std::cout << "-------------------------------------------------------------------------------------------" << std::endl;
    std::cout << " Size of A |    Time    | Procent of nnz in A matrix | Number of values in result matrix " << std::endl;
    std::cout << "-------------------------------------------------------------------------------------------" << std::endl;

    for (size_t i = 0; i < sizesVector.size(); ++i)
    {
        std::cout << "   " << sizesVector[i] << "   |  " << std::fixed << std::setprecision(4) << time[i] << " s  |             " << procentMatrix[i] << "%             |              ";
        std::cout << elementsR[i];
        std::cout << std::endl;
    }

    std::cout << "-------------------------------------------------------------------------------------------" << std::endl;
}

void preformance_test_sequential_with_conversion()
{

    // Tests with larger matrices and different sparsity values
    int size = 10000; // Matrix size for larger tests
    std::cout << "-----------------------------------------" << std::endl;
    std::cout << "Starting performance tests with large matrices (size: " << size << ")..." << std::endl;
    std::cout << "-----------------------------------------" << std::endl;

    std::cout << std::setw(10) << "Sparsity" << " | "
              << std::setw(15) << "Time (Parallel)" << " | "
              << std::setw(15) << "Non-Zero Elements (A)" << " | "
              << std::setw(15) << "Non-Zero Elements (B)" << std::endl;
    std::cout << "-------------------------------------------------------------" << std::endl;

    for (int n = 1; n <= 10; ++n)
    {
        double sparsity = n / 100.0;

        // Generate matrices A and B
        std::cout << "Generating matrix A with sparsity: " << n << "%" << std::endl;
        CRSMatrix A = CRSMatrix::generate_equal(size, sparsity);
        std::cout << "Matrix A generated with " << A.nnz << " non-zero elements." << std::endl;

        std::cout << "Generating matrix B with sparsity: " << n << "%" << std::endl;
        CRSMatrix B = CRSMatrix::generate_random(size, sparsity);
        std::cout << "Matrix B generated with " << B.nnz << " non-zero elements." << std::endl;

        // Measure time for parallel matrix multiplication
        double timeParallel = measure_time([&]()
                                           { DenseMatrix C = A.multiply(B); });

        std::cout << std::setw(10) << n << "%" << " | "
                  << std::setw(15) << timeParallel << " seconds | "
                  << std::setw(15) << A.nnz << " | "
                  << std::setw(15) << B.nnz << std::endl;
    }

    std::cout << "-----------------------------------------" << std::endl;
    std::cout << "Performance tests completed." << std::endl;
}

int main()
{
    std::cout << "Running sequential, implementation preformance test. This test only measures the multiplication of sparse matrixes in CSR format. Running test.." << std::endl;
    preformance_test_sequential_with_conversion();

    std::cout << "Running a sequential variant preformance test. This test measures the multiplication time and conversion of result dense matrix to CSR format! Running test..." << std::endl;
    preformance_test_sequential_without_conversion();

    std::cout << "Running parallel implementation preformance test. This test measures the multiplication time and conversion of result dense matrix to CSR format! Running test..." << std::endl;
    preformance_test_with_conversion();

    std::cout << "Running parallel implementation preformance test. This test only measures the multiplication of sparse matrixes in CSR format. Running test.." << std::endl;
    preformance_test_without_conversion();

    return 0;
}
