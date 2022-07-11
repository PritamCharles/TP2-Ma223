from src.gauss_method import GaussMethod
from src.lu_method import LUMethod
from src.cholesky_method import CholeskyMethod
from src.charts import Chart
import numpy as np
import time


class GetPlots:
    def __init__(self, array_minsize, array_maxsize, step):
        self.gauss = GaussMethod()
        self.LU = LUMethod()
        self.cholesky = CholeskyMethod()
        self.array_minsize = array_minsize
        self.array_maxsize = array_maxsize
        self.step = step

    def get_xplots(self):
        return [i for i in range(self.array_minsize, self.array_maxsize, self.step)]

    def get_yplots_gauss(self):
        timeslist_gauss, normslist_gauss = [], []

        for i in range(self.array_minsize, self.array_maxsize, self.step):
            A = self.cholesky.SDPmatrix(np.random.rand(i, i))
            B = np.random.rand(i, 1)

            start = time.time()
            X = self.gauss.Gauss(A, B)
            end = time.time()
            exectime = end - start
            timeslist_gauss.append(exectime)

            norm = np.linalg.norm(np.dot(A, X) - np.ravel(B))
            normslist_gauss.append(norm)

        return timeslist_gauss, normslist_gauss

    def get_yplots_lu(self):
        timeslist_lu, normslist_lu = [], []

        for i in range(self.array_minsize, self.array_maxsize, self.step):
            A = self.cholesky.SDPmatrix(np.random.rand(i, i))
            B = np.random.rand(i, 1)

            L = self.LU.LUDecomposition(A)[0]
            U = self.LU.LUDecomposition(A)[1]
            start = time.time()
            X = self.LU.LUResolution(L, U, B)
            end = time.time()
            exectime = end - start
            timeslist_lu.append(exectime)

            A = np.dot(L, U)
            norm = np.linalg.norm(np.dot(A, X) - np.ravel(B))
            normslist_lu.append(norm)

        return timeslist_lu, normslist_lu

    def get_yplots_cholesky(self):
        timeslist_cholesky, normslist_cholesky = [], []

        for i in range(self.array_minsize, self.array_maxsize, self.step):
            A = self.cholesky.SDPmatrix(np.random.rand(i, i))
            B = np.random.rand(i, 1)

            start = time.time()
            X = self.cholesky.CholeskyResol(A, B)
            end = time.time()
            exectime = end - start
            timeslist_cholesky.append(exectime)

            norm = np.linalg.norm(np.dot(A, X) - np.ravel(B))
            normslist_cholesky.append(norm)

        return timeslist_cholesky, normslist_cholesky

    def get_yplots_choleskyMac(self):
        timeslist_choleskyMac, normslist_choleskyMac = [], []

        for i in range(self.array_minsize, self.array_maxsize, self.step):
            A = self.cholesky.SDPmatrix(np.random.rand(i, i))
            B = np.random.rand(i, 1)

            start = time.time()
            X = self.cholesky.CholeskyMachineResol(A, B)
            end = time.time()
            exectime = end - start
            timeslist_choleskyMac.append(exectime)

            norm = np.linalg.norm(np.dot(A, X) - np.ravel(B))
            normslist_choleskyMac.append(norm)

        return timeslist_choleskyMac, normslist_choleskyMac

    def get_yplots_choleskyAlt(self):
        timeslist_choleskyAlt, normslist_choleskyAlt = [], []

        for i in range(self.array_minsize, self.array_maxsize, self.step):
            A = self.cholesky.SDPmatrix(np.random.rand(i, i))
            B = np.random.rand(i, 1)

            start = time.time()
            X = self.cholesky.alt_CholeskyResol(A, B)
            end = time.time()
            exectime = end - start
            timeslist_choleskyAlt.append(exectime)

            norm = np.linalg.norm(np.dot(A, X) - np.ravel(B))
            normslist_choleskyAlt.append(norm)

        return timeslist_choleskyAlt, normslist_choleskyAlt

    def get_yplots_np(self):
        timeslist_npsolve, normslist_npsolve = [], []

        for i in range(self.array_minsize, self.array_maxsize, self.step):
            A = self.cholesky.SDPmatrix(np.random.rand(i, i))
            B = np.random.rand(i, 1)

            start = time.time()
            X = np.linalg.solve(A, B)
            end = time.time()
            exectime = end - start
            timeslist_npsolve.append(exectime)

            norm = np.linalg.norm(np.dot(A, X) - B)
            normslist_npsolve.append(norm)

        return timeslist_npsolve, normslist_npsolve


###
plots = GetPlots(array_minsize=3, array_maxsize=353, step=50)

xlist = [plots.get_xplots(), plots.get_xplots(), plots.get_xplots(), plots.get_xplots(), plots.get_xplots(), plots.get_xplots()]
yplots_gauss = [plots.get_yplots_gauss()[0], plots.get_yplots_gauss()[1]]
yplots_lu = [plots.get_yplots_lu()[0], plots.get_yplots_lu()[1]]
yplots_cholesky = [plots.get_yplots_cholesky()[0], plots.get_yplots_cholesky()[1]]
yplots_choleskyMac = [plots.get_yplots_choleskyMac()[0], plots.get_yplots_choleskyMac()[1]]
yplots_choleskyAlt = [plots.get_yplots_choleskyAlt()[0], plots.get_yplots_choleskyAlt()[1]]
yplots_np = [plots.get_yplots_np()[0], plots.get_yplots_np()[1]]
ytimeslist = [yplots_gauss[0], yplots_lu[0], yplots_cholesky[0], yplots_choleskyMac[0], yplots_choleskyAlt[0], yplots_np[0]]
ynormslist = [yplots_gauss[1], yplots_lu[1], yplots_cholesky[1], yplots_choleskyMac[1], yplots_choleskyAlt[1], yplots_np[1]]

labels_list = ["Gauss", "LU", "Cholesky", "Cholesky machine", "Cholesky alternatif", "np.linalg.solve"]
titles_list_gauss = ["Temps d'execution de la methode de Gauss en fonction de la taille de la matrice", "Erreur ||AX - B|| de la methode de Gauss en fonction de la taille de la matrice"]
titles_list_lu = ["Temps d'execution de la methode de LU en fonction de la taille de la matrice", "Erreur ||AX - B|| de la methode LU en fonction de la taille de la matrice"]
titles_list_cholesky = ["Temps d'execution de la methode de Cholesky en fonction de la taille de la matrice", "Erreur ||AX - B|| de la methode Cholesky en fonction de la taille de la matrice"]
titles_list_choleskyMac = ["Temps d'execution de la methode de Cholesky machine en fonction de la taille de la matrice", "Erreur ||AX - B|| de la methode Cholesky machine en fonction de la taille de la matrice"]
titles_list_choleskyAlt = ["Temps d'execution de la methode de Cholesky alteernatif en fonction de la taille de la matrice", "Erreur ||AX - B|| de la methode Cholesky alternatif en fonction de la taille de la matrice"]
titles_list_np = ["Temps d'execution de la methode linalg.solve de Numpy en fonction de la taille de la matrice", "Erreur ||AX - B|| de la methode linalg.solve de Numpy en fonction de la taille de la matrice"]
titles_list_all = ["Temps d'execution des differentes methodes en fonction de la taille de la matrice", "Erreur ||AX - B|| des differentes methodes en fonction de la taille de la matrice"]
axis_labels = ["Taille de la matrice n", "Temps d'execution (en s)", "Erreur ||AX - B||", "log Taille de la matrice n", "log Temps d'execution (en s)"]

# Solo charts
chart1 = Chart(title=titles_list_gauss, alabels=axis_labels)
chart2 = Chart(title=titles_list_lu, alabels=axis_labels)
chart3 = Chart(title=titles_list_cholesky, alabels=axis_labels)
chart4 = Chart(title=titles_list_choleskyMac, alabels=axis_labels)
chart5 = Chart(title=titles_list_choleskyAlt, alabels=axis_labels)
chart6 = Chart(title=titles_list_np, alabels=axis_labels)

chart1.plot(list_xvalues1=xlist[0], list_yvalues1=ytimeslist[0], list_xvalues2=xlist[0], list_yvalues2=ynormslist[0], list_labels=labels_list[0])
chart1.plot_log(list_xvalues=np.log(xlist[0]), list_yvalues=np.log(ytimeslist[0]), list_labels=labels_list[0])

chart2.plot(list_xvalues1=xlist[0], list_yvalues1=ytimeslist[1], list_xvalues2=xlist[0], list_yvalues2=ynormslist[1], list_labels=labels_list[1])
chart2.plot_log(list_xvalues=np.log(xlist[0]), list_yvalues=np.log(ytimeslist[0]), list_labels=labels_list[1])

chart3.plot(list_xvalues1=xlist[0], list_yvalues1=ytimeslist[2], list_xvalues2=xlist[0], list_yvalues2=ynormslist[2], list_labels=labels_list[2])
chart3.plot_log(list_xvalues=np.log(xlist[0]), list_yvalues=np.log(ytimeslist[0]), list_labels=labels_list[2])

chart4.plot(list_xvalues1=xlist[0], list_yvalues1=ytimeslist[3], list_xvalues2=xlist[0], list_yvalues2=ynormslist[3], list_labels=labels_list[3])
chart4.plot_log(list_xvalues=np.log(xlist[0]), list_yvalues=np.log(ytimeslist[0]), list_labels=labels_list[3])

chart5.plot(list_xvalues1=xlist[0], list_yvalues1=ytimeslist[4], list_xvalues2=xlist[0], list_yvalues2=ynormslist[4], list_labels=labels_list[4])
chart5.plot_log(list_xvalues=np.log(xlist[0]), list_yvalues=np.log(ytimeslist[0]), list_labels=labels_list[4])

chart6.plot(list_xvalues1=xlist[0], list_yvalues1=ytimeslist[5], list_xvalues2=xlist[0], list_yvalues2=ynormslist[5], list_labels=labels_list[5])
chart6.plot_log(list_xvalues=np.log(xlist[0]), list_yvalues=np.log(ytimeslist[0]), list_labels=labels_list[5])


# All in one chart
chart7 = Chart(title=titles_list_all, alabels=axis_labels)
chart7.plot_all(nb_plots=6, list_xvalues1=xlist, list_yvalues1=ytimeslist, list_xvalues2=xlist, list_yvalues2=ynormslist, list_labels=labels_list)
