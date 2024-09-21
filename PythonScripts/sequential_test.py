import numpy as np
from scipy.sparse import random, csr_matrix
import time

def sequential_multiply(A, B):
    return A @ B

def generate_sparse_matrix(size, density):
    return random(size, size, density=density, format='csr', dtype=np.float64)

def sequential_test():
    size = 10000
    print(f"Running sequential performance test with matrix size {size}...")
    
    for n in range(1, 11):
        density = n / 100.0
        print(f"Sparsity test: {density * 100}%")

        A = generate_sparse_matrix(size, density)
        B = generate_sparse_matrix(size, density)

        start_time = time.time()
        result = sequential_multiply(A, B)
        end_time = time.time()

        print(f"Time with {density*100}% sparsity: {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    sequential_test()
