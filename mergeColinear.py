import numpy as np

from fitline import fitline
from findSplitPos import findSplitPos

def mergeColinear(xy, alpha, r, pointidx, thresholds):
    z = [alpha(1), r(1)]
    startidx = pointidx(1,1)
    lastendidx = pointidx(1,2)

    N = r.shape[0]
    zt = [0, 0]

    rOut = np.zeros((len(r)),1)
    alphaOut = np.zeros((len(alpha)), 1)
    pointidxOut = np.zeros((len(alpha)), 1)

    j = 0

    for i in range(2, N):
        endidx = pointidx(i,2)

        zt[1], zt[2] = fitline(xy[:, startidx:endidx])

        splitpos = findSplitPos(xy[:, startidx:endidx], zt[1], zt[2], thresholds)

        #Se nao for necessario fazer split, fazemos merge
        if splitpos == -1:
            z = zt
        else: #Sem mais merges
            alphaOut[j, 0] = z[0]
            rOut[j, 0] = z[2]
            pointidxOut[j, :] = [startidx, lastendidx]


        lastendidx = endidx

    #Adicionar o ultimo segmento
    alphaOut[j, 0] = z[0]
    rOut[j, 0] = z[1]
    pointidxOut[j, :] = [startidx, lastendidx]

    return alphaOut, rOut, pointidxOut


def main():
    xy = [[1, 2, 3, 4, 1, 2, 3, 4, 1, 1, 1, 1],
         [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 3, 4]]

    alpha, r = fitline(xy)


if __name__ == '__main__':
    main()




