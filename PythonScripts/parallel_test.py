import numpy as np
from scipy.sparse import random, csr_matrix
import time
from joblib import Parallel, delayed

def parallel_multiply(A, B):
    result = A @ B
    return result

def generate_sparse_matrix(size, density):
    return random(size, size, density=density, format='csr', dtype=np.float64)

def parallel_test():
    size = 10000
    print(f"Running parallel performance test with matrix size {size}...")

    for n in range(1, 11):
        density = n / 100.0
        print(f"Sparsity test: {density * 100}%")

        A = generate_sparse_matrix(size, density)
        B = generate_sparse_matrix(size, density)

        start_time = time.time()

        # Simulate parallel processing by breaking the matrix into chunks
        result = Parallel(n_jobs=-1)(delayed(parallel_multiply)(A, B) for _ in range(1))

        end_time = time.time()

        print(f"Time with {density*100}% sparsity: {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    parallel_test()
