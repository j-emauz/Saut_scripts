from cmath import pi
import sys

import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
#import cv2 as cv


class Thresholds:
    def __init__(self):
        self.seg_min_length = 0.01
        self.point_dist = 0.05
        self.min_point_seg = 1




def fitline(pontos):
    # centroid de pontos considerando que o centroid de 
    # um numero finito de pontos pode ser obtido como 
    # a media de cada coordenada

    lixo, len = pontos.shape
    #alpha = np.zeros((1,1))
    
    xc, yc = pontos.sum(axis = 1) / len
    dx = (pontos[0, :] - xc)
    dy = (pontos[1, :] - yc)

    num = -2 * np.matrix.sum(np.multiply(dx, dy))
    denom = np.matrix.sum(np.multiply(dy, dy) - np.multiply(dx, dx))
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
    #print(splitpos)
    if (splitpos != -1):
        alpha1, r1, idx1 = splitlines(xy, startidx, splitpos+startidx, thresholds) # se calhar start idx-1
        alpha2, r2, idx2 = splitlines(xy, splitpos+startidx, endidx, thresholds)
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

    #rOut = np.zeros((r.shape[0],1))
    #alphaOut = np.zeros((alpha.shape[0], 1))
    #pointidxOut = np.zeros((1, 2))
    rOut = []
    alphaOut = []
    pointidxOut = []


    j = 0

    for i in range(1, N):
        endidx = pointidx[i,1]

        zt[0], zt[1] = fitline(xy[:, startidx:(endidx + 1)])

        splitpos = findsplitpos(xy[:, startidx:(endidx + 1)], zt[0], zt[1], thresholds)


        #Se nao for necessario fazer split, fazemos merge
        if splitpos == -1:
            z = zt
        else: #Sem mais merges
            #alphaOut[j, 0] = z[0]
            alphaOut.append(z[0])
            rOut.append(z[1])
            #rOut[j, 0] = z[1]
            pointidxOut.extend([startidx, lastendidx])
            #pointidxOut = np.vstack((pointidxOut,[startidx, lastendidx]))
            j = j + 1
            z = [alpha[i, 0], r[i, 0]]
            startidx = pointidx[i, 0]


        lastendidx = endidx

    #Adicionar o ultimo segmento
    alphaOut.append(z[0])
    rOut.append(z[1])
    pointidxOut.extend([startidx, lastendidx])

    pointidxOut = np.array(pointidxOut)
    pointidxOut = np.reshape(pointidxOut, (j+1, 2))
    alphaOut = np.array(alphaOut)
    alphaOut = np.reshape(alphaOut, (j + 1, 1))
    rOut = np.array(rOut)
    rOut = np.reshape(rOut, (j+1, 1))

    return alphaOut, rOut, pointidxOut


def pol2cart(theta, rho):
    x = np.zeros((1,theta.shape[0]))
    y = np.zeros((1,theta.shape[0]))
    for i in range(0, theta.shape[0]):
        x[0,i] = rho[i,0] * np.cos(theta[i,0])
        y[0,i] = rho[i,0] * np.sin(theta[i,0])
    return x, y


def extractlines(theta, rho, thersholds):
    # passa de coordenadas polares para cartesianas

    x,y = pol2cart(theta, rho)

    xy = np.vstack((x,y))
    #xy = np.concatenate((x,y),axis=0)
    print(xy)

    startidx =0
    endidx = xy.shape[1] -1 #x e y são vetores linha

    # faz a extracao das linhas
    alpha, r, pointsidx = splitlines(xy, startidx, endidx, thersholds)

    # numero de segmentos de reta, caso seja mais do que um segmento, vereifica se sao colineares
    n = r.shape[0]
    if n > 1:
        alpha, r, pointsidx = mergeColinear(xy, alpha, r, pointsidx, thersholds)
        n = r.shape[0]
        # atualiza o numero de segmentos

    # definir coordenads dos endpoints e len dos segmentos
    segmends = np.zeros((n, 4))
    segmlen = np.zeros((n, 1))
    #for l in range(0, n):
    #    print(np.concatenate([np.transpose(xy[:, pointsidx[l, 0]]), np.transpose(xy[:, pointsidx[l, 1]])], axis = 1))

    for l in range(0, n):
        segmends[l, :] = np.concatenate([np.transpose(xy[:, pointsidx[l, 0]]), np.transpose(xy[:, pointsidx[l, 1]])], axis = 1)
        # segmends[l, :] = [np.transpose(xy[:, pointsidx[l, 0]]), np.transpose(xy[:, pointsidx[l, 1]])]
        #for j in range(0:4):
        #    segmends[l, j] = [xy[j, pointsidx[l, 0]]]
        segmlen[l] = math.sqrt((segmends[l, 0] - segmends[l, 2]) ** 2 + (segmends[l, 1] - segmends[l, 3]) ** 2)
        
    segmlen = np.transpose(segmlen)
    # print(((pointsidx[:,1] - pointsidx[:,0]) >= thersholds.min_point_seg))
    # print((segmlen >= thersholds.seg_min_length))
    # print((segmlen >= thersholds.seg_min_length) & ((pointsidx[:,1] - pointsidx[:,0]) >= thersholds.min_point_seg))

    
    # remover segmentos demasiados pequenos
    #alterar thersholds para params.MIN_SEG_LENGTH e params.MIN_POINTS_PER_SEGMENT
    goodsegmidx = np.argwhere((segmlen >= thersholds.seg_min_length) & ((pointsidx[:,1] - pointsidx[:,0]) >= thersholds.min_point_seg))
    #print(goodsegmidx)
    # goodsegmix2 = goodsegmidx[0, 1]:goodsegmidx[(goodsegmidx.shape[0]), 1]
    # print(goodsegmix2)
    pointsidx = pointsidx[goodsegmidx[:, 1], :]
    #print(pointsidx)
    #print(pointsidx)
    alpha = alpha[goodsegmidx[:, 1], 0]
    r = r[goodsegmidx[:, 1], 0]
    #print(segmends)
    segmends = segmends[goodsegmidx[:, 1], :]
    segmlen = np.transpose(segmlen)
    segmlen = segmlen[goodsegmidx[:, 1], 0]

    #z = np.zeros((alpha.shape[0] - 1, r.shape[0] - 1))
    z = np.transpose(np.vstack(alpha,r))

    R_seg = np.zeros((1, 1, len([len(alpha), 1]) - 1))

    return z, R_seg, segmends




if __name__ == '__main__':
    #pontos = np.matrix([[1, 2, 3, 3, 3], [1, 1, 1, 2, 3]])
    
    #alpha, r = fitline(pontos)
    thresholds = Thresholds()

    """
    alphav, rv, idxv = splitlines(pontos, 0, 8, thresholds)

    print(idxv)

    N = rv.shape[0]
    #print(N)

    if N > 1:
        alphav, rv, idxv = mergeColinear(pontos, alphav, rv, idxv, thresholds)

    """

    # splitpos = findsplitpos(pontos[:, startidx:(endidx + 1)], alpha, r, thresholds)
    #print(alphav)
    #print(rv)
    # print(idxv)
    # print(splitpos)

    theta = np.matrix([[-40*pi/180], [-20*pi/180], [0*pi/180], [20*pi/180], [40*pi/180]])
    rho = np.matrix([[1/math.cos(-40*(pi/180))], [1/math.cos(-20*(pi/180))], [1], [1/math.cos(20*(pi/180))], [1/math.cos(40*(pi/180))]])
    print(theta)
    print(rho)
    z, R_seg, segmends = extractlines(theta, rho, thresholds)
    """
    print(alphav)
    print(rv)
    print(segmends)
    print(segmlen)
    print(pointsidx)
    """
