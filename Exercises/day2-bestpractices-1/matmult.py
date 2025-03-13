import numpy as np
import time  # Import time module for execution timing

N = 250

#@profile
def create_matrix_nn(N):
    """ Create an NxN matrix using NumPy. """
    return np.random.randint(0, 101, (N, N), dtype=np.int32)

#@profile
def create_matrix_nn1(N):
    """ Create an Nx(N+1) matrix using NumPy. """
    return np.random.randint(0, 101, (N, N+1), dtype=np.int32)

#@profile
def matrix_mult_optimized(X, Y):
    """ Optimized matrix multiplication using NumPy's dot product. """
    return np.dot(X, Y)  # NumPy's highly optimized matrix multiplication

# Measure total execution time
start_time = time.time()  # Start timer

# Generate matrices
X = create_matrix_nn(N)
Y = create_matrix_nn1(N)

# Perform optimized matrix multiplication
result = matrix_mult_optimized(X, Y)

end_time = time.time()  # End timer

# Print total execution time
print(f"Total execution time: {end_time - start_time:.6f} seconds")


'''import random
import numpy as np
import time  # Import time module for execution timing

N = 250

#@profile
def create_matrix_nn(N):
    """ Create an NxN matrix. """
    return [[random.randint(0, 100) for _ in range(N)] for _ in range(N)]

#@profile
def create_matrix_nn1(N):
    """ Create an Nx(N+1) matrix. """
    return [[random.randint(0, 100) for _ in range(N+1)] for _ in range(N)]

#@profile
def matrix_mult(X, Y):
    """ Multiply matrices using nested loops. """
    result = [[0] * (N+1) for _ in range(N)]

    for i in range(len(X)):
        for j in range(len(Y[0])):
            for k in range(len(Y)):
                result[i][j] += X[i][k] * Y[k][j]
    
    return result

# Measure total execution time
start_time = time.time()  # Start timer

X = create_matrix_nn(N)
Y = create_matrix_nn1(N)
result = matrix_mult(X, Y)

end_time = time.time()  # End timer

# Print total execution time
print(f"Total execution time: {end_time - start_time:.6f} seconds")'''
