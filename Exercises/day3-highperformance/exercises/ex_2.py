import numpy as np

# a. Create a null vector of size 10 but the fifth value is 1
vector_a = np.zeros(10)
vector_a[4] = 1
print("a:", vector_a)

# b. Create a vector with values ranging from 10 to 49
vector_b = np.arange(10, 50)
print("b:", vector_b)

# c. Reverse a vector
vector_c = vector_b[::-1]
print("c:", vector_c)

# d. Create a 3x3 matrix with values ranging from 0 to 8
matrix_d = np.arange(9).reshape(3, 3)
print("d:\n", matrix_d)

# e. Find indices of non-zero elements
array_e = np.array([1, 2, 0, 0, 4, 0])
indices_e = np.nonzero(array_e)
print("e:", indices_e)

# f. Create a random vector of size 30 and find the mean value
vector_f = np.random.random(30)
mean_f = vector_f.mean()
print("f:", mean_f)

# g. Create a 2D array with 1 on the border and 0 inside
matrix_g = np.ones((5, 5))
matrix_g[1:-1, 1:-1] = 0
print("g:\n", matrix_g)

# h. Create an 8x8 checkerboard matrix
matrix_h = np.zeros((8, 8), dtype=int)
matrix_h[1::2, ::2] = 1
matrix_h[::2, 1::2] = 1
print("h:\n", matrix_h)

# i. Create a checkerboard matrix using tile
matrix_i = np.tile([[0, 1], [1, 0]], (4, 4))
print("i:\n", matrix_i)

# j. Negate all elements between 3 and 8 in a 1D array
Z = np.arange(11)
Z[(Z > 3) & (Z < 8)] *= -1
print("j:", Z)

# k. Create a random vector of size 10 and sort it
vector_k = np.random.random(10)
vector_k.sort()
print("k:", vector_k)

# l. Check if two random arrays are equal
A = np.random.randint(0, 2, 5)
B = np.random.randint(0, 2, 5)
equal_l = np.array_equal(A, B)
print("l:", equal_l)

# m. Square every number in an array in place
Z = np.arange(10, dtype=np.int32)
Z **= 2
print("m:", Z)

# n. Get the diagonal of a dot product
A = np.arange(9).reshape(3, 3)
B = A + 1
C = np.dot(A, B)
D = np.diag(C)
print("n:", D)
