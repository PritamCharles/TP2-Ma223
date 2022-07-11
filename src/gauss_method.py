import numpy as np


class GaussMethod:
    def GaussReduction(self, Aaug):  # Aaug : the augmented matrix of size (n, n+1)
        n, p = np.shape(Aaug)
        for i in range(0, n - 1):
            pivot = i
            if abs(Aaug[i, i]) < 1e-15:  # null pivot verification
                for k in range(i + 1, n):
                    if abs(Aaug[k, k]) > 1e-15:
                        pivot = k
                tmp = np.copy(Aaug[i, :])
                Aaug[i, :] = Aaug[pivot, :]
                Aaug[pivot, :] = tmp

            for j in range(i + 1, n):
                gij = Aaug[j, i] / Aaug[i, i]
                Aaug[j, :] = Aaug[j, :] - (gij * Aaug[i, :])

        return Aaug

    def SysTriSupResolution(self, Taug):  # Taug : the augmented matrix of size (n, n+1)
        n, p = np.shape(Taug)
        X = np.zeros(n)

        for i in range(n - 1, -1, -1):
            somme = 0
            for j in range(i, n):
                somme += X[j] * Taug[i, j]
            X[i] = (Taug[i, -1] - somme) / Taug[i, i]

        return X

    def Gauss(self, A, B):  # A : a matrix of size n ; B : a column matrix
        Aaug = np.c_[
            A, B]  # np.c_[A, B] concatenates the matrix A and the vector B (by transforming B into a vector beforehand)

        self.GaussReduction(Aaug)
        return self.SysTriSupResolution(Aaug)

    def GaussPartialPivotChoice(self, A, B):  # A : a matrix of size n ; B : a column matrix
        Aaug = np.c_[A, B]
        n, p = np.shape(Aaug)

        for i in range(0, n - 1):

            pivot = i
            if abs(Aaug[i, i]) < 1e-15:  # null pivot verification
                for k in range(i + 1, n):
                    if abs(Aaug[k, k]) > 1e-15:
                        pivot = k
                tmp = np.copy(Aaug[i, :])
                Aaug[i, :] = Aaug[pivot, :]
                Aaug[pivot, :] = tmp

            pivot_max = abs(Aaug[i, i])
            indice = i

            for j in range(i + 1, n):
                if abs(Aaug[j, i]) > abs(pivot_max):
                    pivot_max = abs(Aaug[j, i])
                    indice = j

            tmp = np.copy(Aaug[i, :])
            Aaug[i, :] = Aaug[indice, :]
            Aaug[indice, :] = tmp

            A = self.GaussReduction(Aaug)
        return self.SysTriSupResolution(A)

    def GaussTotalPivotChoice(self, A, B):  # A : a matrix of size n ; B : a column matrix
        Aaug = np.c_[A, B]
        n, p = np.shape(Aaug)

        for i in range(0, n - 1):

            pivot = i
            if abs(Aaug[i, i]) < 1e-15:  # null pivot verification
                for k in range(i + 1, n):
                    if abs(Aaug[k, k]) > 1e-15:
                        pivot = k
                tmp = np.copy(Aaug[i, :])
                Aaug[i, :] = Aaug[pivot, :]
                Aaug[pivot, :] = tmp

            for j in range(i, n):
                pivot_max = i

                for elem in range(i + 1, n - 1):
                    if abs(Aaug[j, elem]) > abs(Aaug[j, i]):
                        pivot_max = elem

                tmp = np.copy(Aaug[j, i])
                Aaug[j, i] = Aaug[j, pivot_max]
                Aaug[j, pivot_max] = tmp

        return self.GaussPartialPivotChoice(A, B)