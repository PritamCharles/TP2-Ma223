# -----------------------------------------------------------------------------
# Name:        TP1 - Génie Mathématiques
# Purpose:
#
# Authors:     KANTANE Pritam Charles
# Class:       2PF2
#
# Created:     23/01/2021
# Copyright:
# Licence:     <your licence>
# -----------------------------------------------------------------------------
import numpy as np
import time
import matplotlib.pyplot as plt


#################################################################
# ### Partie 1 ####
#################################################################

# Question 1

def ReductionGauss(Aaug):
    """
        Fonction qui calcule la matrice supérieure d'une matrice augmentée : réduction de Gauss.

        Argument(s):
            - Aaug : une matrice augmentée de taille (n, n+1)

        Retourne:
            La matrice augmentée supérieure.
    """
    n, p = np.shape(Aaug)

    for i in range(0, n - 1):

        pivot = i
        if abs(Aaug[i, i]) < 1e-15:  # Vérification de pivot nul
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


# Question 2

def ResolutionSysTriSup(Taug):
    """
        Fonction qui résout le système d'équation obtenu à partir de la matrice supérieure.

        Argument(s):
            - Taug : une matrice augmentée de taille (n, n+1)

        Retourne:
            Une vecteur colonne qui est le résultat du système linéaire.
    """
    n, p = np.shape(Taug)
    X = np.zeros(n)

    for i in range(n - 1, -1, -1):
        somme = 0
        for j in range(i, n):
            somme += X[j] * Taug[i, j]
        X[i] = (Taug[i, -1] - somme) / Taug[i, i]

    return X


# Question 3

def Gauss(A, B):
    """
        Fonction qui calcule la matrice supérieure et résout le système linéaire obtenu.

        Argument(s):
            - A : une matrice de taille n.
            - B : une matrice colonne.

        Retourne:
             Un vecteur colonne qui est le résultat du système linéaire.
    """
    Aaug = np.c_[A, B]  # np.c_[A, B] concatène la matrice A et le vecteur B (en transformant préalablement B en vecteur)

    ReductionGauss(Aaug)
    return ResolutionSysTriSup(Aaug)


# Question 4
# A la fin du fichier


#################################################################
# ### Partie 2 ####
#################################################################

# Question 1

def DecompositionLU(A):
    """
        Fonction qui calcule la matrice lower de la matrice A.

        Argument(s):
            - A : une matrice de taille n.

        Retourne:
            La matrice Lower L et Upper U.
    """
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

    U = ReductionGauss(A)

    """print("A = L * U où: \n")
    print("U =", "\n", ReductionGauss(A), "\n\n et \n")     # Cette partie permet d'afficher les matrices L et U
    print("L =", "\n", L)"""

    return L, U


# Question 2

def ResolutionLU(L, U, B):
    """
        Fonction qui résout l'équation AX = B où A = LU.

        Argument(s):
            - L: la matrice lower
            - U: la matrice upper
            - B : une matrice colonne

        Retourne:
            Un vecteur colonne qui est le résultat de l'équation AX = B.
    """
    Aaug1 = np.c_[L, B]
    n, p = np.shape(Aaug1)
    Y = np.zeros(n)

    for i in range(0, n):  # résolution par descente

        pivot = i
        if abs(Aaug1[i, i]) < 1e-15:  # Vérification de pivot nul
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

    return Gauss(U, Y)     # résolution par remontée

#################################################################
# ### Partie 3 ####
#################################################################

# Question 1

def GaussChoixPivotPartiel(A, B):
    """
        Fonction qui résout l'équation AX = B par la méthode du pivot partiel.

        Argument(s):
            - A : une matrice de taille n.
            - B : une matrice colonne.


        Retourne:
            Un vecteur colonne qui est le résultat de l'équation AX = B.
    """
    Aaug = np.c_[A, B]
    n, p = np.shape(Aaug)

    for i in range(0, n - 1):

        pivot = i
        if abs(Aaug[i, i]) < 1e-15:  # Vérification de pivot nul
            for k in range(i + 1, n):
                if abs(Aaug[k, k]) > 1e-15:
                    pivot = k
            tmp = np.copy(Aaug[i, :])
            Aaug[i, :] = Aaug[pivot, :]
            Aaug[pivot, :] = tmp

        pivot_max = abs(Aaug[i, i])
        indice = i  # L'indice prend la valeur des i. Il va permettre d'échanger les lignes plus tard

        for j in range(i + 1, n):
            if abs(Aaug[j, i]) > abs(pivot_max):  # on vérifie si le coefficient parcouru est plus grand que le pivot en valeur absolu
                pivot_max = abs(Aaug[j, i])  # si c'est le cas, alors pivot max prend la valeur du coefficient
                indice = j  # j devient le nouvel indice

        tmp = np.copy(Aaug[i, :])
        Aaug[i, :] = Aaug[indice, :]  # Ce processus permet d'échanger deux lignes
        Aaug[indice, :] = tmp

        A = ReductionGauss(Aaug)
    return ResolutionSysTriSup(A)


# Question 2

def GaussChoixPivotTotal(A, B):
    """
        Fonction qui résout l'équation AX = B par la méthode du pivot total.

        Argument(s):
            - A : une matrice de taille n.
            - B : une matrice colonne.


        Retourne:
            Un vecteur colonne qui est le résultat de l'équation AX = B.
    """
    Aaug = np.c_[A, B]
    n, p = np.shape(Aaug)

    for i in range(0, n - 1):

        pivot = i
        if abs(Aaug[i, i]) < 1e-15:  # Vérification de pivot nul
            for k in range(i + 1, n):
                if abs(Aaug[k, k]) > 1e-15:
                    pivot = k
            tmp = np.copy(Aaug[i, :])
            Aaug[i, :] = Aaug[pivot, :]
            Aaug[pivot, :] = tmp

        for j in range(i, n):
            pivot_max = i  # Le pivot max est le coefficient au rang i

            for elem in range(i + 1, n - 1):
                if abs(Aaug[j, elem]) > abs(Aaug[j, i]):  # On vérifie si la valeur absolue de Aaug[j, elem] supérieur aux coefficients au-dessus
                    pivot_max = elem  # Le pivot max prend elem en indice. Cela permettra de changer les colonnes sans soucis

            tmp = np.copy(Aaug[j, i])
            Aaug[j, i] = Aaug[j, pivot_max]  # Processus poour changer deux colonnes
            Aaug[j, pivot_max] = tmp

    return GaussChoixPivotPartiel(A, B)  # On vérifie le pivot maximal sur les lignes et on retourne le résultat


########################################################################
# GRAPHIQUES
########################################################################

# Question 4 (Partie 1) + construction de toutes les courbes


def graphique_LU():
    liste_temps_LU = []
    liste_normes_LU = []
    
    for i in range(3, 503, 50):
        A = np.random.rand(i, i)
        B = np.random.rand(i, 1)

        L = DecompositionLU(A)[0]
        U = DecompositionLU(A)[1]

        debut = time.time()
        ResolutionLU(L, U, B)
        fin = time.time()
        temps_exe_LU = fin - debut
        liste_temps_LU.append(temps_exe_LU)

        X = ResolutionLU(L, U, B)
        A = np.dot(L, U)

        normes_LU = np.linalg.norm(np.dot(A, X) - np.ravel(B))
        liste_normes_LU.append(normes_LU)

    print("graphique LU ...")
    plt.figure(figsize=(15, 9))
    plt.subplot(2, 1, 1)
    x = np.linspace(0, 1003, 10)
    y = np.linspace(0, 1, 3)
    plt.plot(x, liste_temps_LU, label="Méthode de décomposition LU", c="green")
    plt.xlabel("Taille de la matrice n")
    plt.ylabel("Temps d'exécution (en s)")
    plt.title(
        "Graphique représentant le temps d'exécution de la méthode de décomposition LU en fonction de la taille de la matrice")
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.semilogy(x, liste_normes_LU, label="Méthode de décomposition LU", c="green")
    plt.xlabel("Taille de la matrice n")
    plt.ylabel("Erreur ||AX - B||")
    plt.title(
        "Graphique représentant l'erreur ||AX - B|| de la méthode de décomposition LU en fonction de la taille de la matrice")
    plt.legend()
    plt.grid()
    plt.savefig("LU.png")
    plt.show()


def graphique_gauss():
    liste_temps_Gauss = []
    print("temps ...")
    for i in range(3, 1003, 100):
        A = np.random.rand(i, i)
        B = np.random.rand(i, 1)

        debut = time.time()
        Gauss(A, B)
        fin = time.time()
        temps_exe_Gauss = fin - debut
        liste_temps_Gauss.append(temps_exe_Gauss)

    liste_normes_Gauss = []
    print("normes ...")
    for i in range(3, 1003, 100):
        A = np.random.rand(i, i)
        B = np.random.rand(i, 1)

        X = Gauss(A, B)
        normes_Gauss = np.linalg.norm(np.dot(A, X) - np.ravel(B))
        liste_normes_Gauss.append(normes_Gauss)

    print("graphique Gauss ...")
    plt.figure(figsize=(15, 9))
    plt.subplot(2, 1, 1)
    x = np.linspace(0, 1003, 10)
    y = np.linspace(0, 1, 3)
    plt.plot(x, liste_temps_Gauss, label="Méthode de Gauss", c="blue")
    plt.xlabel("Taille de la matrice n")
    plt.ylabel("Temps d'exécution (en s)")
    plt.title(
        "Graphique représentant le temps d'exécution de la méthode de Gauss en fonction de la taille de la matrice")
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.semilogy(x, liste_normes_Gauss, label="Méthode de Gauss", c="blue")
    plt.xlabel("Taille de la matrice n")
    plt.ylabel("Erreur ||AX - B||")
    plt.title(
        "Graphique représentant l'erreur ||AX - B|| de la méthode de Gauss en fonction de la taille de la matrice")
    plt.legend()
    plt.grid()
    plt.savefig("Gauss.png")
    plt.show()


def graphique_pivot_partiel():
    liste_temps_pivot_partiel = []
    print("temps...")
    for i in range(3, 1003, 100):
        print(i)
        A = np.random.rand(i, i)
        B = np.random.rand(i, 1)

        debut = time.time()
        GaussChoixPivotPartiel(A, B)
        fin = time.time()
        temps_exe_pivot_partiel = fin - debut
        liste_temps_pivot_partiel.append(temps_exe_pivot_partiel)

    liste_normes_pivot_partiel = []
    print("normes...")
    for i in range(3, 1003, 100):
        A = np.random.rand(i, i)
        B = np.random.rand(i, 1)

        X = GaussChoixPivotPartiel(A, B)
        normes_pivot_partiel = np.linalg.norm(np.dot(A, X) - np.ravel(B))
        liste_normes_pivot_partiel.append(normes_pivot_partiel)

    print("graphique pivot partiel ...")
    plt.figure(figsize=(15, 9))
    plt.subplot(2, 1, 1)
    x = np.linspace(0, 1003, 10)
    y = np.linspace(0, 1, 10)
    plt.plot(x, liste_temps_pivot_partiel, label="Algorithme du pivot partiel", c="red")
    plt.xlabel("Taille de la matrice n")
    plt.ylabel("Temps d'exécution (en s)")
    plt.title(
        "Graphique représentant le temps d'exécution de l'algorithme du pivot partiel en fonction de la taille de la matrice")
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.semilogy(x, liste_normes_pivot_partiel, label="Algorithme du pivot partiel", c="red")
    plt.xlabel("Taille de la matrice n")
    plt.ylabel("Erreur ||AX - B||")
    plt.title(
        "Graphique représentant l'erreur ||AX - B|| de l'algorithme du pivot partiel en fonction de la taille de la matrice")
    plt.legend()
    plt.grid()
    plt.savefig("pivot_partiel.png")
    plt.show()


def graphique_pivot_total():
    liste_temps_pivot_total = []
    print("temps ...")

    for i in range(3, 1003, 100):
        A = np.random.rand(i, i)
        B = np.random.rand(i, 1)
        debut = time.time()
        GaussChoixPivotTotal(A, B)
        fin = time.time()
        temps_exe_pivot_total = fin - debut
        liste_temps_pivot_total.append(temps_exe_pivot_total)

    liste_normes_pivot_total = []
    print("normes ...")

    for i in range(3, 1003, 100):
        A = np.random.rand(i, i)
        B = np.random.rand(i, 1)

        X = GaussChoixPivotTotal(A, B)
        normes_pivot_total = np.linalg.norm(np.dot(A, X) - np.ravel(B))
        liste_normes_pivot_total.append(normes_pivot_total)

    print("graphique pivot total ...")
    plt.figure(figsize=(15, 9))
    plt.subplot(2, 1, 1)
    x = np.linspace(0, 1003, 10)
    y = np.linspace(0, 1, 10)
    plt.plot(x, liste_temps_pivot_total, label="Algorithme du pivot total", c="yellow")
    plt.xlabel("Taille de la matrice n")
    plt.ylabel("Temps d'exécution (en s)")
    plt.title(
        "Graphique représentant le temps d'exécution de l'algorithme du pivot total en fonction de la taille de la matrice")
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.semilogy(x, liste_normes_pivot_total, label="Algorithme du pivot total", c="yellow")
    plt.xlabel("Taille de la matrice n")
    plt.ylabel("Erreur ||AX - B||")
    plt.title(
        "Graphique représentant l'erreur ||AX - B|| de l'algorithme du pivot total en fonction de la taille de la matrice")
    plt.legend()
    plt.grid()
    plt.savefig("pivot_total.png")
    plt.show()


def graphique_linalgsolve():
    liste_temps_linalgsolve = []
    print("temps ...")

    for i in range(3, 1003, 100):
        A = np.random.rand(i, i)
        B = np.random.rand(i, 1)
        debut = time.time()
        np.linalg.solve(A, B)
        fin = time.time()
        temps_exe_linalgsolve = fin - debut
        liste_temps_linalgsolve.append(temps_exe_linalgsolve)

    liste_normes_linalgsolve = []

    for i in range(3, 1003, 100):
        print("normes linalgsolve n =", i)
        A = np.random.rand(i, i)
        B = np.random.rand(i, 1)

        X = np.linalg.solve(A, B)
        normes_linalgsolve = np.linalg.norm(np.dot(A, X) - B)
        liste_normes_linalgsolve.append(normes_linalgsolve)

    print("graphique linalgsolve ...")
    plt.figure(figsize=(15, 9))
    plt.subplot(2, 1, 1)
    x = np.linspace(0, 1003, 10)
    y = np.linspace(0, 1, 10)
    plt.plot(x, liste_temps_linalgsolve, label="np.linalg.solve", c="black")
    plt.xlabel("Taille de la matrice n")
    plt.ylabel("Temps d'exécution (en s)")
    plt.title(
        "Graphique représentant le temps d'exécution de la méthode linalg.solve de Numpy en fonction de la taille de la matrice")
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.semilogy(x, liste_normes_linalgsolve, label="np.linalg.solve", c="black")
    plt.xlabel("Taille de la matrice n")
    plt.ylabel("Erreur ||AX - B||")
    plt.title(
        "Graphique représentant l'erreur ||AX - B|| de la méthode linalg.solve de Numpy en fonction de la taille de la matrice")
    plt.legend()
    plt.grid()
    plt.savefig("linalg.solve.png")
    plt.show()


def graphique_all():
    liste_temps_LU = []
    for i in range(3, 503, 50):
        A = np.random.rand(i, i)
        B = np.random.rand(i, 1)
        L = DecompositionLU(A)[0]
        U = DecompositionLU(A)[1]

        print("temps LU n =", i)
        debut = time.time()
        ResolutionLU(L, U, B)
        fin = time.time()
        temps_exe_LU = fin - debut
        liste_temps_LU.append(temps_exe_LU)

    liste_normes_LU = []
    for i in range(3, 503, 50):
        Aa = np.random.rand(i, i)
        B = np.random.rand(i, 1)
        L = DecompositionLU(Aa)[0]
        U = DecompositionLU(Aa)[1]
        X = ResolutionLU(L, U, B)
        A = np.dot(L, U)

        print("normes LU n =", i)
        normes_LU = np.linalg.norm(np.dot(A, X) - np.ravel(B))
        liste_normes_LU.append(normes_LU)

    liste_temps_Gauss = []
    for i in range(3, 503, 50):
        A = np.random.rand(i, i)
        B = np.random.rand(i, 1)

        print("temps Gauss n =", i)
        debut = time.time()
        Gauss(A, B)
        fin = time.time()
        temps_exe_Gauss = fin - debut
        liste_temps_Gauss.append(temps_exe_Gauss)

    liste_normes_Gauss = []
    for i in range(3, 503, 50):
        A = np.random.rand(i, i)
        B = np.random.rand(i, 1)

        print("normes Gauss n =", i)
        X = Gauss(A, B)
        normes_Gauss = np.linalg.norm(np.dot(A, X) - np.ravel(B))
        liste_normes_Gauss.append(normes_Gauss)

    liste_temps_pivot_partiel = []
    for i in range(3, 503, 50):
        print(i)
        A = np.random.rand(i, i)
        B = np.random.rand(i, 1)

        print("temps pivot partiel n =", i)
        debut = time.time()
        GaussChoixPivotPartiel(A, B)
        fin = time.time()
        temps_exe_pivot_partiel = fin - debut
        liste_temps_pivot_partiel.append(temps_exe_pivot_partiel)

    liste_normes_pivot_partiel = []
    for i in range(3, 503, 50):
        A = np.random.rand(i, i)
        B = np.random.rand(i, 1)

        print("normes pivot partiel n =", i)
        X = GaussChoixPivotPartiel(A, B)
        normes_pivot_partiel = np.linalg.norm(np.dot(A, X) - np.ravel(B))
        liste_normes_pivot_partiel.append(normes_pivot_partiel)

    liste_temps_pivot_total = []
    for i in range(3, 503, 50):
        A = np.random.rand(i, i)
        B = np.random.rand(i, 1)

        print("temps pivot total n =", i)
        debut = time.time()
        GaussChoixPivotTotal(A, B)
        fin = time.time()
        temps_exe_pivot_total = fin - debut
        liste_temps_pivot_total.append(temps_exe_pivot_total)

    liste_normes_pivot_total = []
    for i in range(3, 503, 50):
        A = np.random.rand(i, i)
        B = np.random.rand(i, 1)

        print("normes pivot total n =", i)
        X = GaussChoixPivotTotal(A, B)
        normes_pivot_total = np.linalg.norm(np.dot(A, X) - np.ravel(B))
        liste_normes_pivot_total.append(normes_pivot_total)

    liste_temps_linalgsolve = []
    for i in range(3, 503, 50):
        A = np.random.rand(i, i)
        B = np.random.rand(i, 1)

        print("temps linalg.solve n=", i)
        debut = time.time()
        np.linalg.solve(A, B)
        fin = time.time()
        temps_exe_linalgsolve = fin - debut
        liste_temps_linalgsolve.append(temps_exe_linalgsolve)

    liste_normes_linalgsolve = []
    for i in range(3, 503, 50):
        print("normes linalgsolve n =", i)
        A = np.random.rand(i, i)
        B = np.random.rand(i, 1)

        X = np.linalg.solve(A, B)
        normes_linalgsolve = np.linalg.norm(np.dot(A, X) - B)
        liste_normes_linalgsolve.append(normes_linalgsolve)

    plt.figure(figsize=(15, 9))
    plt.subplot(2, 1, 1)
    x = np.linspace(0, 503, 10)
    y = np.linspace(0, 1, 10)
    plt.plot(x, liste_temps_Gauss, label="Gauss", c="blue")
    plt.plot(x, liste_temps_LU, label="LU", c="green")
    plt.plot(x, liste_temps_pivot_partiel, label="Algorithme pivot partiel", c="red")
    plt.plot(x, liste_temps_pivot_total, label="Algorithme pivot total", c="violet")
    plt.plot(x, liste_temps_linalgsolve, label="np.linalg.solve", c="black")
    plt.xlabel("Taille de la matrice n")
    plt.ylabel("Temps d'exécution (en s)")
    plt.title(
        "Graphique représentant le temps d'exécution des algorithmes en fonction de la taille de la matrice")
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.semilogy(x, liste_normes_Gauss, label="Gauss", c="blue")
    plt.semilogy(x, liste_normes_LU, label="LU", c="green")
    plt.semilogy(x, liste_normes_pivot_partiel, label="Algorithme pivot partiel", c="red")
    plt.semilogy(x, liste_normes_pivot_total, label="Algorithme pivot total", c="yellow")
    plt.plot(x, liste_normes_linalgsolve, label="np.linalg.solve", c="black")
    plt.xlabel("Taille de la matrice n")
    plt.ylabel("Erreur ||AX - B||")
    plt.title(
        "Graphique représentant l'erreur ||AX - B|| des algorithmes en fonction de la taille de la matrice")
    plt.legend()
    plt.grid()
    plt.savefig("toutes_les_méthodes.png")
    plt.show()


def graphique_all_loglog():
    liste_temps_LU = []
    for i in range(3, 303, 100):
        A = np.random.rand(i, i)
        B = np.random.rand(i, 1)
        L = DecompositionLU(A)[0]
        U = DecompositionLU(A)[1]

        debut = time.time()
        ResolutionLU(L, U, B)
        fin = time.time()
        temps_exe_LU = fin - debut
        liste_temps_LU.append(temps_exe_LU)

    liste_temps_Gauss = []
    for i in range(3, 303, 100):
        A = np.random.rand(i, i)
        B = np.random.rand(i, 1)

        debut = time.time()
        Gauss(A, B)
        fin = time.time()
        temps_exe_Gauss = fin - debut
        liste_temps_Gauss.append(temps_exe_Gauss)

    liste_temps_pivot_partiel = []

    for i in range(3, 303, 100):
        A = np.random.rand(i, i)
        B = np.random.rand(i, 1)

        debut = time.time()
        GaussChoixPivotPartiel(A, B)
        fin = time.time()
        temps_exe_pivot_partiel = fin - debut
        liste_temps_pivot_partiel.append(temps_exe_pivot_partiel)

    liste_temps_pivot_total = []

    for i in range(3, 303, 100):
        A = np.random.rand(i, i)
        B = np.random.rand(i, 1)
        debut = time.time()
        GaussChoixPivotTotal(A, B)
        fin = time.time()
        temps_exe_pivot_total = fin - debut
        liste_temps_pivot_total.append(temps_exe_pivot_total)

    plt.figure(figsize=(15, 9))

    x = np.linspace(0, 303, 3)
    y = np.linspace(0, 1, 3)
    # plt.plot(np.log(x), np.log(liste_temps_Gauss), label="Gauss", c="blue")
    plt.plot(np.log(x), np.log(liste_temps_LU), label="LU", c="green")
    # plt.plot(np.log(x), np.log(liste_temps_pivot_partiel), label="Algorithme pivot partiel", c="red")
    # plt.plot(np.log(x), np.log(liste_temps_pivot_total), label="Algorithme pivot total", c="violet")
    plt.xlabel("log Taille de la matrice n")
    plt.ylabel("log Temps d'exécution (en s)")
    plt.title(
        "Graphique représentant le temps d'exécution de la décomposition LU en fonction de la taille de la matrice")
    plt.legend()
    plt.grid()
    plt.savefig("toutes_les_méthodes_log_log.png")
    plt.show()


#############################################

# graphique_gauss()
#graphique_LU()
# graphique_pivot_partiel()
# graphique_pivot_total()
# graphique_linalgsolve()
# graphique_all()
# graphique_all_loglog()
