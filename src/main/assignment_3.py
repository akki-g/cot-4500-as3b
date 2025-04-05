import numpy as np


def backwards_substitution(A):
    n = len(A)
    A = np.array(A, dtype=float)

    for i in range(n):
        for k in range(i +1, n):
            factor = A[k][i] / A[i][i]
            A[k][i:] -= factor * A[i][i:]


    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (A[i][-1] - np.dot(A[i][i + 1:n], x[i + 1:n])) / A[i][i]

    return np.array(x, dtype=int)


def LU_factorization(A):
    n = len(A)
    A = np.array(A, dtype=float)
    L = np.identity(n)
    U = np.copy(A)

    for i in range(n):
        for j in range(i+1, n):
            factor = U[j][i] / U[i][i]
            L[j][i] = factor
            U[j][i:] -= factor * U[i][i:]


    return L, U


def diagonal_dominance(A):
    n = len(A)
    A = np.array(A, dtype=float)
    for i in range(n):
        diag = abs(A[i][i])
        row_sum = sum(abs(A[i][j]) for j in range(n) if j != i)
        if diag <= row_sum:
            return False
    return True


def is_positive_definite(A):

    if not np.allclose(A, A.T):
        return False
    
    eigenvalues = np.linalg.eigvals(A)
    return np.all(eigenvalues > 0)