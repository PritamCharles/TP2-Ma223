import matplotlib.pyplot as plt
import itertools


class Chart:
    def __init__(self, title, alabels):
        self.title = title
        self.axis_labels = alabels

    def plot_log(self, list_xvalues, list_yvalues, list_labels):
        plt.figure(figsize=(15, 9))
        plt.plot(list_xvalues, list_yvalues, label=list_labels)
        plt.title(self.title[0])
        plt.xlabel(self.axis_labels[3])
        plt.ylabel(self.axis_labels[4])
        plt.legend()
        plt.grid()
        plt.show()

    def plot(self, list_xvalues1, list_yvalues1, list_labels, list_xvalues2=None, list_yvalues2=None):
        plt.figure(figsize=(15, 9))

        plt.subplot(2, 1, 1)
        plt.plot(list_xvalues1, list_yvalues1, label=list_labels)
        plt.title(self.title[0])
        plt.xlabel(self.axis_labels[0])
        plt.ylabel(self.axis_labels[1])
        plt.legend()
        plt.grid()

        plt.subplot(2, 1, 2)
        plt.semilogy(list_xvalues2, list_yvalues2, label=list_labels)
        plt.title(self.title[1])
        plt.xlabel(self.axis_labels[0])
        plt.ylabel(self.axis_labels[2])
        plt.legend()
        plt.grid()

        plt.show()

    def plot_all(self, list_xvalues1, list_yvalues1, list_labels, nb_plots=1, list_xvalues2=None, list_yvalues2=None):
        plt.figure(figsize=(15, 9))

        plt.subplot(2, 1, 1)
        for i, j, k, l in itertools.zip_longest(range(nb_plots), range(nb_plots), range(nb_plots),
                                                range(nb_plots), fillvalue=""):
            plt.plot(list_xvalues1[j], list_yvalues1[k], label=list_labels[l])

        plt.title(self.title[0])
        plt.legend()
        plt.xlabel(self.axis_labels[0])
        plt.ylabel(self.axis_labels[1])
        plt.grid()

        plt.subplot(2, 1, 2)
        for i, j, k, l in itertools.zip_longest(range(nb_plots), range(nb_plots), range(nb_plots),
                                                range(nb_plots), fillvalue=""):
            plt.semilogy(list_xvalues2[j], list_yvalues2[k], label=list_labels[l])

        plt.title(self.title[1])
        plt.legend()
        plt.xlabel(self.axis_labels[0])
        plt.ylabel(self.axis_labels[2])
        plt.grid()

        plt.show()
