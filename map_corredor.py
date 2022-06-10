import matplotlib.pyplot as plt
import numpy as np

P11 = np.array([-3, -3])
P12 = np.array([3, -3])
X1 = [P11[0], P12[0]]
Y1 = [P11[1], P12[1]]

P21 = np.array([-3, -1.5])
P22 = np.array([1.5, -1.5])
X2 = [P21[0], P22[0]]
Y2 = [P21[1], P22[1]]

P31 = np.array([3, -3])
P32 = np.array([3, 3])
X3 = [P31[0], P32[0]]
Y3 = [P31[1], P32[1]]

""" quando usa a parte do elevador, fica em comentario
P41 = np.array([1.5, -1.5])
P42 = np.array([1.5, 3])
X4 = [P41[0], P42[0]]
Y4 = [P41[1], P42[1]]
"""
#parte elevador
P51 = np.array([-1.5, 0])
P52 = np.array([1.5, 0])
X5 = [P51[0], P52[0]]
Y5 = [P51[1], P52[1]]

P61 = np.array([-1.5, 0])
P62 = np.array([-1.5, 2])
X6 = [P61[0], P62[0]]
Y6 = [P61[1], P62[1]]

P71 = np.array([-1.5, 2])
P72 = np.array([1.5, 2])
X7 = [P71[0], P72[0]]
Y7 = [P71[1], P72[1]]

P41 = np.array([1.5, -1.5])
P42 = np.array([1.5, 0])
X4 = [P41[0], P42[0]]
Y4 = [P41[1], P42[1]]

P81 = np.array([1.5, 2])
P82 = np.array([1.5, 3])
X8 = [P81[0], P82[0]]
Y8 = [P81[1], P82[1]]

def plot_map():
    plt.plot(X1, Y1, '-k')
    plt.plot(X2, Y2, '-k')
    plt.plot(X3, Y3, '-k')
    plt.plot(X4, Y4, '-k')
    #parte do elevador
    plt.plot(X5, Y5, '-k')
    plt.plot(X6, Y6, '-k')
    plt.plot(X7, Y7, '-k')
    plt.plot(X8, Y8, '-k')

if __name__ == '__main__':
    plt.gcf().canvas.mpl_connect('key_release_event',
                                 lambda event: [exit(0) if event.key == 'escape' else None])
    plot_map()

    plt.show()