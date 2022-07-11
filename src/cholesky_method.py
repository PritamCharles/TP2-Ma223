import numpy as np


class CholeskyMethod:
    def SDPmatrix(self, M):
        if np.linalg.det(M) != 0:
            A = np.dot(np.transpose(M), M)
            return A
        else:
            print("La matrice n'est pas inversible.")

    def Cholesky(self, A):
        n, m = np.shape(A)
        L = np.zeros((n, n))
        n, p = np.shape(L)

        for k in range(n):
            somme1 = 0
            for j in range(0, k):
                somme1 += L[k, j] ** 2
            L[k, k] = np.sqrt(A[k, k] - somme1)

            for i in range(k + 1, n):
                somme2 = 0
                for j in range(0, k):
                    somme2 += L[i, j] * L[k, j]
                L[i, k] = (A[i, k] - somme2) / L[k, k]

        return L

    def SystTriSupResolution(self, Taug):
        n, m = np.shape(Taug)
        x = np.zeros(n)

        for k in range(n - 1, -1, -1):
            somme = 0
            for i in range(k, n):
                somme = somme + Taug[k, i] * x[i]
            x[k] = (Taug[k, -1] - somme) / Taug[k, k]

        return x

    def SystTriInfResolution(self, Taug):
        n, m = np.shape(Taug)
        x = np.zeros((n, 1))

        for k in range(n):
            somme = 0
            for i in range(k):
                somme = somme + Taug[k, i] * x[i, 0]
            x[k, 0] = (Taug[k, -1] - somme) / Taug[k, k]  # -1 car derniere colonne

        return x

    def CholeskyResol(self, A, B):
        n, m = np.shape(A)
        L = self.Cholesky(A)
        Laug = np.concatenate((L, B), axis=1)

        Y = np.zeros(n)
        Y = self.SystTriInfResolution(Laug)

        Yaug = np.concatenate((L.T, Y), axis=1)
        X = self.SystTriSupResolution(Yaug)

        return X

    def CholeskyMachineResol(self, A, B):
        n, m = np.shape(A)
        L = np.linalg.cholesky(A)
        Laug = np.concatenate((L, B), axis=1)

        Y = np.zeros(n)
        Y = self.SystTriInfResolution(Laug)

        Yaug = np.concatenate((L.T, Y), axis=1)
        X = self.SystTriSupResolution(Yaug)

        return X

    def alt_Cholesky(self, A):
        n, p = np.shape(A)
        L = np.eye(n)
        D = np.eye(n)
        n, p = np.shape(L)

        for k in range(n):
            somme1 = 0
            for j in range(0, k):
                somme1 += L[k, j] ** 2 * D[j, j]
            D[k, k] = A[k, k] - somme1

            for i in range(k + 1, n):
                somme2 = 0
                for j in range(0, k):
                    somme2 += L[i, j] * L[k, j] * D[j, j]
                L[i, k] = (1 / D[k, k]) * (A[i, k] - somme2)

        return L, D

    def alt_CholeskyResol(self, A, B):
        n, m = np.shape(A)
        L = self.alt_Cholesky(A)[0]
        D = self.alt_Cholesky(A)[1]
        Laug = np.c_[(L, B)]

        Y = np.zeros(n)
        Y = self.SystTriInfResolution(Laug)

        Yaug = np.c_[(D, Y)]
        Z = self.SystTriSupResolution(Yaug)

        Zaug = np.c_[(np.transpose(L), Z)]
        X = self.SystTriSupResolution(Zaug)

        return X
