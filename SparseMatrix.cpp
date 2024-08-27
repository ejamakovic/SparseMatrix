#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>
#include <thread>
#include <omp.h>
#include <algorithm>
#include <numeric>

template <typename T>
class CSC_Format {
private:
    struct Node {
        T value;
        int row;
        Node* next;

        Node(int r, T v, Node* n = nullptr) : row(r), value(v), next(n) {}
    };

    Node** nodesBegin; 
    Node** nodesEnd; 
    int* col_pointers; 
    int* col_noe;
    int rows;
    int cols;
    std::atomic<int> nnz{ 0 };
    T zero = 0;

public:
    CSC_Format(int r, int c) : rows(r), cols(c) {
        nodesBegin = new Node * [cols] {};  
        nodesEnd = new Node * [cols] {}; 
        col_pointers = new int[cols + 1]();
        col_noe = new int[cols]();
    }

    ~CSC_Format() {


        delete[] col_pointers;
        delete[] col_noe;

        for (int i = 0; i < cols; ++i) {
            Node* current = nodesBegin[i];
            while (current) {
                Node* next = current->next;
                delete current;
                current = next;
            }
        }
        delete[] nodesBegin;
        delete[] nodesEnd;
    }

    void addElement(int row, int col, T value) {
        Node* newNode = new Node(row, value, nullptr);

        if (!nodesBegin[col]) {
            
            nodesBegin[col] = newNode;
            nodesEnd[col] = newNode;  
        }
        else {
            
            Node* current = nodesBegin[col];
            Node* prev = nullptr;

            while (current && row > current->row) {
                prev = current;
                current = current->next;
            }

            if (prev && prev->row == row) {
                prev->value = value;
                return;
            }

            if (!prev) {

                
                newNode->next = current;
                nodesBegin[col] = newNode;  
            }
            else {

                
                prev->next = newNode;
                newNode->next = current;

                
                if (!current) {
                    nodesEnd[col] = newNode;  
                }
            }
        }

        col_noe[col]++;  
        nnz++;
    }


    
    void addElementForward(int row, int col, T value) {
        Node* newNode = new Node(row, value, nullptr);

        if (!nodesBegin[col]) {
            
            nodesBegin[col] = newNode;
            nodesEnd[col] = newNode;  
        }
        else {
            
            Node* current = nodesEnd[col];
            current->next = newNode;
            nodesEnd[col] = newNode;
        }

        col_noe[col]++;  
        nnz++;
    }


    const T& GetElement(int row, int col) const {
        if ((nodesBegin[col] && row < nodesBegin[col]->row) || (nodesEnd[col] && row > nodesEnd[col]->row))
            return zero;
        Node* current = nodesBegin[col];
        while (current) {
            if (current->row == row)
                return current->value;
            if (current->row > row)
                return zero;
            current = current->next;
        }
        return zero;
    }

    int getNNZ() const {
        return nnz;
    }

    const int* getColPointers() const {
        for (int i = 1; i <= rows; ++i) {
            col_pointers[i] = col_pointers[i - 1] + col_noe[i - 1];
        }
        return col_pointers;
    }

    std::pair<T*, int> GetColumn(int col) {
        T* values = new T[rows]();
        Node* current = nodesBegin[col];
        while (current) {
            values[current->row] = current->value;
            current = current->next;

        }
        return std::make_pair(values, col_noe[col]);
    }


    void printColPointers() const {
        getColPointers();
        std::cout << "Col Pointers: ";
        for (int i = 0; i <= cols; ++i) {
            std::cout << col_pointers[i] << " ";
        }
        std::cout << std::endl;
    }

    void printNodesBeginEnd() const {
        for (int i = 0; i < cols; i++) {
            auto node = nodesBegin[i];
            if (node) {
                auto end = nodesEnd[i];
                std::cout << "Pocetak reda " << i << " kolona " << node->row << " vrijednost " << node->value << " kraj reda kolona " << end->row << " vrijednost " << end->value << std::endl;
            }
        }
    }
};


template <typename T>
class CSR_Format {
private:
    struct Node {
        T value;
        int col;
        Node* next;

        Node(int c, T v, Node* n = nullptr) : col(c), value(v), next(n) {}
    };

    Node** nodesBegin; 
    Node** nodesEnd; 
    int* row_pointers; 
    int* row_noe;
    int rows;
    int cols;
    int nnz = 0;
    T zero = 0;

public:
    CSR_Format(int r, int c) : rows(r), cols(c) {
        nodesBegin = new Node * [rows] {};  
        nodesEnd = new Node * [rows] {}; 
        row_pointers = new int[rows + 1]();
        row_noe = new int[rows]();
    }

    ~CSR_Format() {


        delete[] row_pointers;
        delete[] row_noe;

        for (int i = 0; i < rows; ++i) {
            Node* current = nodesBegin[i];
            while (current) {
                Node* next = current->next;
                delete current;
                current = next;
            }
        }
        delete[] nodesBegin;
        delete[] nodesEnd;
    }

    void addElement(int row, int col, T value) {
        Node* newNode = new Node(col, value, nullptr);       

        if (!nodesBegin[row]) {           
            nodesBegin[row] = newNode;
            nodesEnd[row] = newNode; 
        }
        else {            
            Node* current = nodesBegin[row];
            Node* prev = nullptr;

            while (current && col > current->col) {
                prev = current;
                current = current->next;
            }

            if (prev && prev->col == col) {
                prev->value = value;
                return;
            }

            if (!prev) {
                newNode->next = current;
                nodesBegin[row] = newNode;  
            }
            else {

                
                prev->next = newNode;
                newNode->next = current;

                
                if (!current) {
                    nodesEnd[row] = newNode;  
                }
            }
        }

        row_noe[row]++;  
        nnz++;
    }

    
    void addElementForward(int row, int col, T value) {
        Node* newNode = new Node(col, value, nullptr);

        if (!nodesBegin[row]) {
            
            nodesBegin[row] = newNode;
            nodesEnd[row] = newNode;  
        }
        else {
            
            Node* current = nodesEnd[row];
            current->next = newNode;
            nodesEnd[row] = newNode;
        }

        row_noe[row]++; 
        nnz++;
    }


    const T& GetElement(int row, int col) const {
        if ((nodesBegin[row] && col < nodesBegin[row]->col) || (nodesEnd[row] && col > nodesEnd[row]->col))
            return zero;
        Node* current = nodesBegin[row];
        while (current) {
            if (current->col == col)
                return current->value;
            if (current->col > col)
                return zero;
            current = current->next;
        }
        return zero;
    }




    int getNNZ() const {
        return nnz;
    }

    const int* getRowPointers() const {
        for (int i = 1; i <= rows; ++i) {
            row_pointers[i] = row_pointers[i - 1] + row_noe[i - 1];
        }
        return row_pointers;
    }

    T* GetMatrix() {
        T* m = new T[rows * cols]();

#pragma omp parallel for
        for (int i = 0; i < rows; i++) {
            Node* current = nodesBegin[i];
            while (current) {
                m[i * rows + current->col] = current->value;
                current = current->next;
            }
        }

        return m;
    }


    std::pair<T*, int*> GetAllElementsWithColumns() {

        T* values = new T[nnz];
        int* indexColumns = new int[nnz];
        int index = 0;

        for (int i = 0; i < rows; i++) {
            Node* current = nodesBegin[i];            
            while (current) {
                values[index] = current->value;
                indexColumns[index++] = current->col;
                current = current->next;
            }
        }
        return std::make_pair(values, indexColumns);
    }



    void printRowPointers() const {
        getRowPointers();
        std::cout << "Row Pointers: ";
        for (int i = 0; i <= rows; ++i) {
            std::cout << row_pointers[i] << " ";
        }
        std::cout << std::endl;
    }

    void printNodesBeginEnd() const {
        for (int i = 0; i < rows; i++) {
            auto node = nodesBegin[i];
            if (node) {
                auto end = nodesEnd[i];
                std::cout << "Pocetak reda " << i << " kolona " << node->col << " vrijednost " << node->value << " kraj reda kolona " << end->col << " vrijednost " << end->value << std::endl;
            }
        }
    }
};

template <typename T> class SparseMatrix {
private:
    int rows;
    int cols;
    CSR_Format<T>* csr;
    CSC_Format<T>* csc;

public:
    SparseMatrix(int rows, int cols) : rows(rows), cols(cols) {
        csr = new CSR_Format<T>(rows, cols);
        csc = new CSC_Format<T>(rows, cols);
    }

    ~SparseMatrix() {
        delete csr;
        delete csc;
    }

    void addElement(int row, int col, T value) {
        if (row < 0 || row >= rows || col < 0 || col >= cols) {
            std::cout << "Ne moze se dodati element sa tim indkesom u matricu." << std::endl;
            return;
        }
        csr->addElement(row, col, value);
        csc->addElement(row, col, value);
    }

    void addElementForward(int row, int col, T value, int m = 0) {
        if (row < 0 || row >= rows || col < 0 || col >= cols) {
            std::cout << "Ne moze se dodati element sa tim indkesom u matricu." << std::endl;
            return;
        }
        csc->addElementForward(row, col, value);
        if (!m)
            csr->addElementForward(row, col, value);
    }

    int getNNZ() {
        return csc->getNNZ();
    }

    SparseMatrix<T>* multParallel(const SparseMatrix<T>& matrixB) const {
        if (cols != matrixB.rows) {
            std::cout << "Ne mogu se pomnoziti ove matrice." << std::endl;
            SparseMatrix<T>* r = new SparseMatrix<T>(0, 0);
            return r;
        }

        int row = rows;
        int col = matrixB.cols;

        const auto& rpA = csr->getRowPointers();
        const auto& A = csr->GetAllElementsWithColumns();
        const auto& valuesA = A.first;
        const auto& colsA = A.second;


        SparseMatrix<T>* result = new SparseMatrix<T>(row, col);
#pragma omp parallel for 
        for (int j = 0; j < col; ++j)
        {
            // Izvuci cijelu kolonu iz matrice B
            const auto& pair = matrixB.csc->GetColumn(j);
            const auto& colB = pair.first;
            if (pair.second == 0) {
                delete[] colB;
                continue;
            }            
#pragma omp parallel for
            for (int i = 0; i < rows; ++i) {

                T dotProduct = 0;
                // Racunaj skalarni produkt retka iz prve matrice i kolone iz druge matrice
                const int& nextRow = rpA[i + 1];

#pragma omp simd reduction(+:dotProduct)
                for (int k = rpA[i]; k < nextRow; ++k) {
                    const auto& valueB = colB[colsA[k]];
                    if (valueB != 0)
                        dotProduct += valuesA[k] * valueB;
                }

                if (dotProduct != 0) {

                    // Dodaj rezultat u rezultantnu matricu
                    result->addElementForward(i, j, dotProduct, 1);

                }

            }
            delete[] colB;
        }

        return result;
    }





    SparseMatrix<T>* mult(const SparseMatrix<T>& matrixB) const {
        if (cols != matrixB.rows) {
            std::cout << "Ne mogu se pomnoziti ove matrice." << std::endl;
            SparseMatrix<T>* r = new SparseMatrix<T>(0, 0);
            return r;
        }

        int row = rows;
        int col = matrixB.cols;
        const auto& rpA = csr->getRowPointers();
        const auto& A = csr->GetAllElementsWithColumns();
        const auto& valuesA = A.first;
        const auto& colsA = A.second;


        SparseMatrix<T>* result = new SparseMatrix<T>(row, col);

        for (int j = 0; j < col; ++j)
        {
            // Izvuci cijelu kolonu iz matrice B
            const auto& pair = matrixB.csc->GetColumn(j);
            const auto& colB = pair.first;
            if (pair.second == 0) {
                delete[] colB;
                continue;
            }
            for (int i = 0; i < rows; ++i) {
                T dotProduct = 0;

                // Racunaj skalarni produkt retka iz prve matrice i kolone iz druge matrice
                const int& nextRow = rpA[i + 1];

                for (int k = rpA[i]; k < nextRow; ++k) {
                    const auto& valueB = colB[colsA[k]];
                    if (valueB != 0)
                        dotProduct += valuesA[k] * valueB;
                }

                if (dotProduct != 0)
                    // Dodaj rezultat u rezultantnu matricu
                    result->addElementForward(i, j, dotProduct, 1);
            }
            delete[] colB;
        }
        return result;
    }


    const T& GetElement(int row, int col) const {
        if (row < 0 || row >= rows || col < 0 || col >= cols) {
            throw std::out_of_range("Pristup izvan opsega matrice");
        }
        return csc->GetElement(row, col);
    }



    void print() {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                const auto a = csc->GetElement(i, j);
                std::cout << a << " ";
            }
            std::cout << std::endl;
        }
    }

    const int& getRows() {
        return rows;
    }

    const int& getCols() {
        return cols;
    }

};


// Pravi matricu tako da su 1 samo onoliko koliko ih treba biti po rijetkosti
template <typename T>
void dodajElementeURedove(SparseMatrix<T>& matrica, int n) {
    int rows = matrica.getRows();
    int cols = matrica.getCols();
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < n; ++j) {
            matrica.addElementForward(i, j, 1);
        }
    }
}

// Pravi matricu da je svaka pozicija u redu random
template <typename T>
void dodajElementeURedove2(SparseMatrix<T>& matrica, double percentage) {
    int rows = matrica.getRows();
    int cols = matrica.getCols();
    int totalElements = static_cast<int>(rows * cols * percentage / 100.0);

    for (int i = 0; i < rows; ++i) {

        std::vector<int> indices(cols);
        std::iota(indices.begin(), indices.end(), 0);

        std::random_shuffle(indices.begin(), indices.end());

        int elementsToExtract = totalElements / rows;
        std::vector<int> extractedIndices(indices.begin(), indices.begin() + elementsToExtract);
        for (int j = 0; j < totalElements / rows; ++j) {
            matrica.addElement(i, indices[j], 1);
        }
    }
}


int main() {
    int startSize = 10000;
    int endSize = 10000;
    std::vector<int> sizesVector;
    std::vector<double> sequentialForwardTimes;
    std::vector<double> parallelForwardTimes;
    std::vector<int> elementsRow;
    std::vector<int> procentMatrix;
    for (int n = 1; n <= 1; n += 5) {
        if (n == 6) n = 5;
        for (int size = startSize; size <= endSize; size *= 10) {
            sizesVector.push_back(size);
            procentMatrix.push_back(n);
            SparseMatrix<int> m1(size, size);
            dodajElementeURedove2(m1, n);            

            auto startSequentialForward = std::chrono::high_resolution_clock::now();
            //auto sequentialForwardResult = m1.mult(m1);
            //delete sequentialForwardResult;
            auto endSequentialForward = std::chrono::high_resolution_clock::now();
            double sequentialForwardTime = std::chrono::duration<double>(endSequentialForward - startSequentialForward).count();

            int repeat = 1;
            double average = 0;

            for (int k = 1; k <= repeat; k++) {
                auto startParallelForward = std::chrono::high_resolution_clock::now();
                auto parallelForwardResult = m1.multParallel(m1);
                auto endParallelForward = std::chrono::high_resolution_clock::now();
                double parallelForwardTime = std::chrono::duration<double>(endParallelForward - startParallelForward).count();
                average += parallelForwardTime;
                elementsRow.push_back(parallelForwardResult->getNNZ());
                delete parallelForwardResult;
            }

            sequentialForwardTimes.push_back(sequentialForwardTime);
            parallelForwardTimes.push_back(average / repeat);
        }
        std::cout << "Gotovo mnozenje za n = " << n << std::endl;
    }

    // Ispis rezultata
    std::cout << "-----------------------------------------------------" << std::endl;
    std::cout << " Rows | Seq. Forward | Par. Forward | Procent of nnz " << std::endl;
    std::cout << "-----------------------------------------------------" << std::endl;

    for (size_t i = 0; i < sizesVector.size(); ++i) {
        std::cout << sizesVector[i] << "   | "
            << sequentialForwardTimes[i] << " s       | " << parallelForwardTimes[i] << " s       | " << procentMatrix[i] << "% | " << elementsRow[i] << std::endl;
    }

    std::cout << "--------------------------------------------------------" << std::endl;

    return 0;

}