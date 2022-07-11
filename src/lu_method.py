import numpy as np
import src.gauss_method as gm


class LUMethod:
    def __init__(self):
        self.gauss = gm.GaussMethod()

    def LUDecomposition(self, A):  # A : matrix of size n
        n, p = np.shape(A)
        L = np.eye(n)

        for i in range(0, n - 1):
            for j in range(i + 1, n):
                gij = A[j, i] / A[i, i]
                L[j, i] = gij
                A[j, :] = A[j, :] - (gij * A[i, :])

        for i in range(0, n - 1):
            for j in range(i + 1, n):
                gij = A[j, i] / A[i, i]
                A[j, :] = A[j, :] - (gij * A[i, :])

        U = self.gauss.GaussReduction(A)

        return L, U

    def LUResolution(self, L, U, B):  # L: the lower matrix ; U: the upper matrix ; B : a column matrix
        Aaug1 = np.c_[L, B]
        n, p = np.shape(Aaug1)
        Y = np.zeros(n)

        for i in range(0, n):  # descent resolution

            pivot = i
            if abs(Aaug1[i, i]) < 1e-15:  # null pivot verification
                for k in range(i + 1, n):
                    if abs(Aaug1[k, k]) > 1e-15:
                        pivot = k
                tmp = np.copy(Aaug1[i, :])
                Aaug1[i, :] = Aaug1[pivot, :]
                Aaug1[pivot, :] = tmp

            somme = 0
            for j in range(-1, n - 1):
                somme += Y[j] * Aaug1[i, j]
            Y[i] = (Aaug1[i, -1] - somme)

        return self.gauss.Gauss(U, Y)  # upward resolution
