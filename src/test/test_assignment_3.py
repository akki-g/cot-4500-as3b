import numpy as np
from src.main.assignment_3 import *

A = np.array([
    [2, -1, 1, 6],
    [1, 3, 1, 0],
    [-1, 5, 4, -3]
])

res = backwards_substitution(A)
print(res)


A = np.array([
    [1, 1, 0, 3],
    [2, 1, -1, 1],
    [3, -1, -1, 2],
    [-1, 2, 3, -1]
])

L, U = LU_factorization(A)

det = np.linalg.det(A)
print('\n', det)
print('\n', L)
print('\n', U)


A = np.array([
    [9, 0, 5, 2, 1],
    [3, 9, 1, 2, 1],
    [0, 1, 7, 2, 3],
    [4, 2, 3, 12, 2],
    [3, 2, 4, 0, 8]
])

res = diagonal_dominance(A)
if res:
    print(f"\n {res} The matrix is diagonally dominant")
else:
    print(f"\n {res} The matrix is not diagonally dominant")



A = np.array([
    [2, 2, 1],
    [2, 3, 0],
    [1, 0, 2]
])

res = is_positive_definite(A)
if res:
    print(f"\n {res} The matrix is positive definite")
else:
    print(f"\n {res} The matrix is not positive definite")
