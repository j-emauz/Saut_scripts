from cmath import pi
import sys

import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv


class Thresholds:
    def __init__(self):
        self.seg_min_length = 0.01
        self.point_dist = 0.005
        self.min_point_seg = 20




def fitline(pontos):
    # centroid de pontos considerando que o centroid de 
    # um numero finito de pontos pode ser obtido como 
    # a media de cada coordenada

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


def compdistpointstoline(xy, alpha, r):
    xcosa = xy[1, :] * math.cos(alpha)
    ysina = xy[2, :] * math.sin(alpha)
    d = xcosa + ysina - r
    return d


def findsplitposid(d, thresholds):
    # implementaçao simples
    N = len(d)

    d = abs(d)
    mask = d > thresholds.point_dist
    if not np.any(mask):
        splitpos = -1
        return splitpos
    
    splitpos = np.argmax(d)
    if (splitpos == 0):
        splitpos = 1
        return splitpos
    if(splitpos == (N-1)):
        splitpos = N-2
        return splitpos




def findsplitpos(xy, alpha, r, thresholds):
    d  = compdistpointstoline(xy, alpha, r)
    splitpos = findsplitposid(d, thresholds)
    return splitpos


def splitlines(xy, startidx, endidx, thresholds):
    N = endidx - startidx + 1

    alpha, r = fitline(xy[:, startidx:(endidx + 1)])

    if N <= 2:
        idx = [startidx, endidx]
        return alpha, r, idx

    splitpos = findsplitpos(xy[:, startidx:(endidx + 1)], alpha, r, thresholds)
    if (splitpos != -1):
        alpha1, r1, idx1 = splitlines(xy, startidx, splitpos+startidx-1, thresholds) # se calhar start idx-1
        alpha2, r2, idx2 = splitlines(xy, splitpos+startidx-1, endidx, thresholds)
        alpha = np.concatenate(alpha1, alpha2)
        r = np.concatenate(r1, r2)
        idx = np.concatenate(idx1, idx2)
    else:
        idx = [startidx, endidx]

    return alpha, r, idx
    



if __name__ == '__main__':
    pontos = np.matrix([[1, 2, 2, 3, 5, 4], [4, 3, 2, 4 ,2, 5]])
    
    alpha, r = fitline(pontos)
    thresholds = Thresholds()

    print(alpha)
    print(r)
