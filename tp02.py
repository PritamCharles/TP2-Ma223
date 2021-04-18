###################################################################################
# TP02 - Ma223 - Résolution de systèmes linéaires par la méthode de Cholesky
# Auteur : Pritam Charles Kantane - 2PF2
# Date : 31/03/2021
###################################################################################
import numpy as np
import tp01_vf as tp01
from matplotlib import pyplot as plt
import time
#from simpy import linalg

def matriceSDP(M):
    if np.linalg.det(M) != 0:
        A = np.dot(np.transpose(M), M)
        return A
    else:
        print("La matrice n'est pas inversible.")
# Partie 1
# Question 1


def Cholesky(A):
    n, m = np.shape(A)
    L = np.zeros((n, n))
    n, p = np.shape(L)

    for k in range(n):
        somme1 = 0
        for j in range(0, k):
            somme1 += L[k, j]**2
        L[k, k] = np.sqrt(A[k, k] - somme1)

        for i in range(k + 1, n):
            somme2 = 0
            for j in range(0, k):
                somme2 += L[i, j] * L[k, j]
            L[i, k] = (A[i, k] - somme2) / L[k, k]

    Lt = np.transpose(L)

    return L

# A = np.array([[4, -2, -4], [-2, 10, 5], [-4, 5, 6]])
# print("décomposition de A:\n", "L =\n", Cholesky(A)[0], ",\n\n", " LT =\n", Cholesky(A)[1])

# Partie 2
# Question 1


def ResolutionSystTriSup(Taug):
    # print("taille=",np.shape(Taug))
    n, m = np.shape(Taug)
    x = np.zeros(n)
    for k in range(n-1, -1, -1):  # n-1 par rapport au nombre de lignes;
        # -1 car on a besoin de la position 0;
        # -1 pour avoir un pas qui permet de remonter.
        somme = 0
        for i in range(k, n):
            somme = somme + Taug[k, i]*x[i]
        x[k] = (Taug[k, -1]-somme)/Taug[k, k]  # -1 car derniere colonne
    # print("x=","\n",x)
    return x


def ResolutionSystTriInf(Taug):
    # print("taille=",shape(Taug))
    n, m = np.shape(Taug)
    x = np.zeros((n, 1))

    for k in range(n):
        somme = 0
        for i in range(k):
            somme = somme + Taug[k, i]*x[i, 0]
        x[k, 0] = (Taug[k, -1]-somme)/Taug[k, k]  # -1 car derniere colonne

    # print("x=","\n",x)
    return x


def ResolCholesky(A, B):
    n, m = np.shape(A)
    L = Cholesky(A)
    Laug = np.concatenate((L, B), axis=1)

    Y = np.zeros(n)
    Y = ResolutionSystTriInf(Laug)

    Yaug = np.concatenate((L.T, Y), axis=1)
    X = ResolutionSystTriSup(Yaug)
    return X

def ResolCholeskyMachine(A, B):
	n,m = np.shape(A)
	L = np.linalg.cholesky(A)
	Laug = np.concatenate((L, B), axis=1)

	Y = np.zeros(n)
	Y = ResolutionSystTriInf(Laug)

	Yaug = np.concatenate((L.T, Y), axis=1)
	X = ResolutionSystTriSup(Yaug)
	return X
# Méthode de Cholesky alternatif

def Cholesky_alt(A):
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


def ResolCholesky_alt(A, B):
    n, m = np.shape(A)
    L = Cholesky_alt(A)[0]
    D = Cholesky_alt(A)[1]
    Laug = np.c_[(L, B)]
    Y = np.zeros(n)
    Y = ResolutionSystTriInf(Laug)
    # print('Y=',Y)
    # print('Lt=',L.T)

    Yaug = np.c_[(D, Y)]
    # print('Zaug=',Zaug)
    Z = ResolutionSystTriSup(Yaug)

    Zaug = np.c_[(np.transpose(L), Z)]
    # print('Yaug=',Yaug)
    X = ResolutionSystTriSup(Zaug)
    return X

def scipylu(A, B):
    n, p = np.shape(A)
    X = np.zeros(n)
    
    L = linalg.lu(A)[1]
    U = linalg.lu(A)[2]
    tp01.ResolutionLU(L, U ,B)

    return X

########################################################################
# GRAPHIQUES
########################################################################

# Construction de toutes les courbes

def graph_all():
    liste_temps_Cholesky = []
    liste_normes_Cholesky = []

    liste_temps_Solve = []
    liste_normes_Solve = []

    liste_temps_Gauss = []
    liste_normes_Gauss = []

    liste_temps_cholesky_machine = []
    liste_normes_cholesky_machine = []

    liste_temps_LU = []
    liste_normes_LU = []

    liste_temps_Cholesky_alt = []
    liste_normes_Cholesky_alt = []

    #liste_temps_LU_machine = []
    #liste_normes_LU_machine = []

    for i in range(3, 503, 50):
        A = matriceSDP(np.random.rand(i, i))
        B = np.random.rand(i, 1)

        #---------------Cholesky----------------
        debut_cholesky = time.time()
        Cholesky = ResolCholesky(A, B)
        fin_cholesky = time.time()

        temps_exe_Cholesky = fin_cholesky - debut_cholesky
        liste_temps_Cholesky.append(temps_exe_Cholesky)

        normes_Cholesky = np.linalg.norm(np.dot(A, Cholesky) - np.ravel(B))
        liste_normes_Cholesky.append(normes_Cholesky)

        #---------------Solve----------------        
        debut_solve = time.time()
        Solve = np.linalg.solve(A, B)
        fin_solve = time.time()

        temps_exe_solve = fin_solve - debut_solve
        liste_temps_Solve.append(temps_exe_solve)

        normes_solve = np.linalg.norm(np.dot(A, Solve) - B)
        liste_normes_Solve.append(normes_solve)


        #---------------Gauss----------------
        debut_gauss = time.time()
        Gauss = tp01.Gauss(A, B)
        fin_gauss = time.time()

        temps_exe_gauss = fin_gauss - debut_gauss
        liste_temps_Gauss.append(temps_exe_gauss)

        normes_gauss = np.linalg.norm(np.dot(A, Gauss) - np.ravel(B))
        liste_normes_Gauss.append(normes_gauss)        
        
        #---------------Cholesky Machine----------------
        debut_cholesky_machine = time.time()
        CholeskyMachine = ResolCholeskyMachine(A, B)
        fin_cholesky_machine = time.time()

        temps_exe_cholesky_machine = fin_cholesky_machine - debut_cholesky_machine
        liste_temps_cholesky_machine.append(temps_exe_cholesky_machine)

        normes_cholesky_machine = np.linalg.norm(np.dot(A, CholeskyMachine) - np.ravel(B))
        liste_normes_cholesky_machine.append(normes_cholesky_machine)

        
        #---------------LU----------------

        L = tp01.DecompositionLU(A)[0]
        U = tp01.DecompositionLU(A)[1]

        debut = time.time()
        tp01.ResolutionLU(L, U, B)
        fin = time.time()
        temps_exe_LU = fin - debut
        liste_temps_LU.append(temps_exe_LU)

        X = tp01.ResolutionLU(L, U, B)
        A = np.dot(L, U)

        normes_LU = np.linalg.norm(np.dot(A, X) - np.ravel(B))
        liste_normes_LU.append(normes_LU)

         # ---------------Cholesky ALternatif----------------

        debut_cholesky_alt = time.time()
        CholeskyAlt = ResolCholesky_alt(A, B)
        fin_cholesky_alt = time.time()

        temps_exe_cholesky_alt = fin_cholesky_alt - debut_cholesky_alt
        liste_temps_Cholesky_alt.append(temps_exe_cholesky_alt)

        normes_cholesky_alt = np.linalg.norm(np.dot(A, CholeskyAlt) - np.ravel(B))
        liste_normes_Cholesky_alt.append(normes_cholesky_alt)

        #---------------LU Machine----------------
        """
        debut_LU = time.time()
        LU = scipylu(A, B)
        fin_LU = time.time()

        temps_exe_LU = fin_LU- debut_LU
        liste_temps_LU.append(temps_exe_LU)

        normes_LU = np.linalg.norm(np.dot(A, LU) - np.ravel(B))
        liste_normes_LU.append(normes_LU)         
        """

    plt.figure(figsize=(15, 9))
    plt.subplot(2, 1, 1)
    x = np.linspace(0, 503, 10)
    y = np.linspace(0, 10, 3)
    plt.plot(x, liste_temps_Cholesky, label="Méthode de Cholesky", c="red")
    plt.plot(x, liste_temps_Cholesky_alt, label="Cholesky Alternatif", c="magenta")
    plt.plot(x, liste_temps_cholesky_machine, label="Cholesky Machine", c="black")
    plt.plot(x, liste_temps_Solve, label="Méthode linalg.solve", c="lime")
    plt.plot(x, liste_temps_Gauss, label="Méthode de Gauss", c="cyan")
    plt.plot(x, liste_temps_LU, label="LU", c="orange")
    #plt.plot(x, liste_temps_LU_machine, label="LU", c="purple")
    
    
    plt.xlabel("Taille de la matrice n")
    plt.ylabel("Temps d'exécution (en s)")
    plt.title(
        "Graphique représentant le temps d'exécution en fonction de la taille de la matrice")
    plt.legend()
    plt.grid()


    plt.subplot(2, 1, 2)
    plt.semilogy(x, liste_normes_Cholesky, label="Méthode de Cholesky", c="red")
    plt.semilogy(x, liste_normes_Cholesky_alt, label="Cholesky Alternatif", c="magenta")
    plt.semilogy(x, liste_normes_cholesky_machine, label="Cholesky Machine", c="black")
    plt.semilogy(x, liste_normes_Solve, label="Méthode linalg.solve", c="green")
    plt.semilogy(x, liste_normes_Gauss, label="Méthode de Gauss", c="cyan")
    plt.semilogy(x, liste_normes_LU, label="LU", c="orange")
    #plt.semilogy(x, liste_normes_LU_machine, label='LU Machine', c='purple')
    
    plt.xlabel("Taille de la matrice n")
    plt.ylabel("Erreur ||AX - B||")
    plt.title(
        "Graphique représentant l'erreur ||AX - B|| des méthodes en fonction de la taille de la matrice")
    plt.legend()
    plt.grid()
    plt.savefig("graph.png")
    plt.show()


graph_all()
