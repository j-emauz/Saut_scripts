from cmath import pi
import sys

import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

def fitline(pontos):

    lixo, len = pontos.shape
    
    xc, yc = pontos.sum(axis = 1) / len
    dx = (pontos[0, :] - xc)
    dy = (pontos[1, :] - yc)

    num = -2 * np.matrix.sum(np.multiply(dx, dy))
    denom = np.matrix.sum(np.multiply(dy, dy) - np.multiply(dx, dx))
    alpha = math.atan2(num, denom) / 2

    r = xc * math.cos(alpha) * yc * math.sin(alpha)

    if r < 0:
        alpha = alpha + math.pi
        if alpha > pi:
            alpha = alpha - 2 * math.pi
        r = -r


    return alpha, r



if __name__ == '__main__':
    pontos = np.matrix([[1, 2], [4, 3]])
    
    lixo, len = fitline(pontos)

    print(lixo)
    print(len)