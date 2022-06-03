import numpy as np

from cmath import pi
import sys

import math
import matplotlib
import matplotlib.pyplot as plt
#import cv2 as cv

class Thresholds:
    def __init__(self):
        self.seg_min_length = 0.01
        self.point_dist = 0.05
        self.min_point_seg = 20


def fitline(pontos):
    # centroid de pontos considerando que o centroid de
    # um numero finito de pontos pode ser obtido como
    # a media de cada coordenada

    lixo, len = pontos.shape
    # alpha = np.zeros((1,1))

    xc, yc = pontos.sum(axis=1) / len
    dx = (pontos[0, :] - xc)
    dy = (pontos[1, :] - yc)

    num = -2 * np.sum(np.multiply(dx, dy))
    denom = np.sum(np.multiply(dy, dy) - np.multiply(dx, dx))
    alpha = math.atan2(num, denom) / 2

    r = xc * math.cos(alpha) + yc * math.sin(alpha)

    if r < 0:
        alpha = alpha + math.pi
        if alpha > pi:
            alpha = alpha - 2 * math.pi
        r = -r

    return alpha, r


def compdistpointstoline(xy, alpha, r):
    xcosa = xy[0, :] * math.cos(alpha)
    ysina = xy[1, :] * math.sin(alpha)
    d = xcosa + ysina - r
    return d


def findsplitposid(d, thresholds):
    # implementaçao simples
    # print('d = ', end = '')
    # print(d)
    N = d.shape[1]
    # print('N =', end='')
    # print(N)

    d = abs(d)
    # print(d)
    mask = d > thresholds.point_dist
    # print('mask =', end='')
    # print(mask)
    if not np.any(mask):
        splitpos = -1
        return splitpos

    splitpos = np.argmax(d)
    # print(splitpos)
    if (splitpos == 0):
        splitpos = 1
    if (splitpos == (N - 1)):
        splitpos = N - 2
    return splitpos


def findsplitpos(xy, alpha, r, thresholds):
    d = compdistpointstoline(xy, alpha, r)
    splitpos = findsplitposid(d, thresholds)
    return splitpos


def splitlines(xy, startidx, endidx, thresholds):
    N = endidx - startidx + 1

    alpha, r = fitline(xy[:, startidx:(endidx + 1)])

    if N <= 2:
        idx = [startidx, endidx]
        return alpha, r, idx

    splitpos = findsplitpos(xy[:, startidx:(endidx + 1)], alpha, r, thresholds)
    # print(splitpos)
    if (splitpos != -1):
        alpha1, r1, idx1 = splitlines(xy, startidx, splitpos + startidx, thresholds)  # se calhar start idx-1
        alpha2, r2, idx2 = splitlines(xy, splitpos + startidx, endidx, thresholds)
        alpha = np.vstack((alpha1, alpha2))
        r = np.vstack((r1, r2))
        idx = np.vstack((idx1, idx2))
    else:
        idx = np.array([startidx, endidx])

    return alpha, r, idx


def mergeColinear(xy, alpha, r, pointidx, thresholds):
    z = [alpha[0, 0], r[0, 0]]
    startidx = pointidx[0, 0]
    lastendidx = pointidx[0, 1]

    N = r.shape[0]
    zt = [0, 0]

    # rOut = np.zeros((r.shape[0],1))
    # alphaOut = np.zeros((alpha.shape[0], 1))
    # pointidxOut = np.zeros((1, 2))
    rOut = []
    alphaOut = []
    pointidxOut = []

    j = 0

    for i in range(1, N):
        endidx = pointidx[i, 1]

        zt[0], zt[1] = fitline(xy[:, startidx:(endidx + 1)])

        splitpos = findsplitpos(xy[:, startidx:(endidx + 1)], zt[0], zt[1], thresholds)

        # Se nao for necessario fazer split, fazemos merge
        if splitpos == -1:
            z = zt
        else:  # Sem mais merges
            # alphaOut[j, 0] = z[0]
            alphaOut.append(z[0])
            rOut.append(z[1])
            # rOut[j, 0] = z[1]
            print('Dentro do else')
            print(startidx)
            print(lastendidx)
            pointidxOut.extend([startidx, lastendidx])
            # pointidxOut = np.vstack((pointidxOut,[startidx, lastendidx]))
            j = j + 1
            z = [alpha[i, 0], r[i, 0]]
            startidx = pointidx[i, 0]

        lastendidx = endidx

    # Adicionar o ultimo segmento
    alphaOut.append(z[0])
    rOut.append(z[1])
    pointidxOut.extend([startidx, lastendidx])

    pointidxOut = np.array(pointidxOut)
    pointidxOut = np.reshape(pointidxOut, (j + 1, 2))
    alphaOut = np.array(alphaOut)
    alphaOut = np.reshape(alphaOut, (j + 1, 1))
    rOut = np.array(rOut)
    rOut = np.reshape(rOut, (j + 1, 1))

    return alphaOut, rOut, pointidxOut


if __name__ == '__main__':
    pontos = np.array([[1, 1, 1, 1, 1], [-0.83909963, -0.36397023, 0.0, 0.36397023, 0.83909963]])
    print(pontos)


    alpha, r = fitline(pontos)
    print(alpha)
    print(r)

    '''
    thresholds = Thresholds()
    alphav = np.array([[1.57079633],[1.57079633]])
    rv = np.array([[1],[1]])

    idxv = np.array([[0, 3], [3, 7]])

    alphav, rv, idxv = mergeColinear(pontos, alphav, rv, idxv, thresholds)


    #for i in range (1,N):
    #    segend(i,:) = [XY(:, pointIdx(i, 1))' XY(:, pointIdx(i,2))'];

    print(idxv)
    '''