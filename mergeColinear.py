from fitline import fitline
from findSplitPos import findSplitPos

def mergeColinear(xy, alpha, r, pointidx, thresholds):
    z = [alpha(1), r(1)]
    startidx = pointidx(1,1)
    lastendidx = pointidx(1,2)

    N = r.shape[0]
    zt = [0, 0]

    rOut = [0, 0]
    alphaOut = [0, 0]
    pointidxOut = [0, 0]
    j = 0

    for i in range(2, N):
        endidx = pointidx(i,2)

        zt[1], zt[2] = fitline(xy[:, startidx:endidx])

        splitpos = findSplitPos(xy[:, startidx:endidx], zt[1], zt[2], thresholds)

        #Se nao for necessario fazer split, fazemos merge
        if splitpos == -1
            z = zt
        elif #Sem mais merges
            alphaOut[j, 0] = z[0]
            rOut[j, 0] = z[2]
            pointidxOut[j, :] = [startidx, lastendidx]


        lastendidx = endidx

    #Adicionar o ultimo segmento
    alphaOut[j, 0] = z[0]
    rOut[j, 0] = z[1]
    pointidxOut[j, :] = [startidx, lastendidx]

    return alphaOut, rOut, pointidxOut




