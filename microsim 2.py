import sys

from cmath import pi
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

"""
x1 = np.linspace(-3, 3, 100)
y1 = 0 * x1 - 3
y2 = 0 * x1 + 3

y3 = np.linspace(-3, 3, 100)
x2 = -3 + 0 * y3
x3 = 3 + 0 * y3
"""
# linha de baixo
P11 = np.array([-3, -3])
P12 = np.array([3, -3])
X1 = [P11[0], P12[0]]
Y1 = [P11[1], P12[1]]
# linha de cima
P21 = np.array([-3, 3])
P22 = np.array([3, 3])
X2 = [P21[0], P22[0]]
Y2 = [P21[1], P22[1]]
# linha da esquerda
P31 = np.array([-3, -3])
P32 = np.array([-3, 3])
X3 = [P31[0], P32[0]]
Y3 = [P31[1], P32[1]]
# linha da direita
P41 = np.array([3, -3])
P42 = np.array([3, 3])
X4 = [P41[0], P42[0]]
Y4 = [P41[1], P42[1]]
#linha la po meio
P51 = np.array([1,-2.5])
P52 = np.array([2.5,0])
X5 = [P51[0], P52[0]]
Y5 = [P51[1], P52[1]]

INPUT_NOISE = np.diag([0.01, np.deg2rad(0.5)])  # ** 2
SIM_TIME = 0
DT = 0.1

R = np.diag([
    0.1,  # variance of location on x-axis
    0.1,  # variance of location on y-axis
    np.deg2rad(1.0),  # variance of yaw angle
]) ** 2  # predict state covariance


def plot_map():
    plt.plot(X1, Y1, '-k')
    plt.plot(X2, Y2, '-k')
    plt.plot(X3, Y3, '-k')
    plt.plot(X4, Y4, '-k')
    plt.plot(X5, Y5, '-k')
    #plt.show()



def ekf_estimation(xEst, Eest, u):
    # Predict step
    # xEst é o anterior e vai ser atualizado no final
    G_x = np.array([[1.0, 0, -u[0, 0] * math.sin(xEst[2, 0] + u[1, 0])],
                    [0, 1.0, u[0, 0] * math.cos(xEst[2, 0] + u[1, 0])],
                    [0, 0, 1.0]])

    b = np.array([[u[0, 0] * math.cos(xEst[2, 0] + u[1, 0])],
                  [u[0, 0] * math.sin(xEst[2, 0] + u[1, 0])],
                  [u[1, 0]]])

    Eest = G_x @ Eest @ G_x.T + R
    xEst = xEst + b

    return xEst, Eest


def observation(xTrue, xDR, u):
    b = np.array([[u[0, 0] * math.cos(xTrue[2, 0] + u[1, 0])],
                  [u[0, 0] * math.sin(xTrue[2, 0] + u[1, 0])],
                  [u[1, 0]]])
    xTrue = xTrue + b

    ud = u + INPUT_NOISE @ np.random.randn(2, 1)
    bd = np.array([[ud[0, 0] * math.cos(xDR[2, 0] + ud[1, 0])],
                   [ud[0, 0] * math.sin(xDR[2, 0] + ud[1, 0])],
                   [ud[1, 0]]])
    xDR = xDR + bd

    return xTrue, xDR, ud

"""
def perp( a ) :
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

def seg_intersect(a1,a2, b1,b2) :
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = np.dot( dap, db)
    num = np.dot( dap, dp )
    if denom.astype(float) == 0:                          # lines are parallel
        return float('inf'), float('inf')
    return (num / denom.astype(float))*db + b1
"""

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def get_intersect(p11, p12, p21, p22):
    """ 
    Returns the point of intersection of the lines passing through p12,p11 and p22,p21.
    p11: [x, y] a point on the first line
    p12: [x, y] another point on the first line
    p21: [x, y] a point on the second line
    p22: [x, y] another point on the second line
    """
    s = np.vstack([p11,p12,p21,p22])        # s for stacked
    h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
    l1 = np.cross(h[0], h[1])           # get first line
    l2 = np.cross(h[2], h[3])           # get second line
    x, y, z = np.cross(l1, l2)          # point of intersection
    if z == 0:                          # lines are parallel
        return float('inf'), float('inf')
    return x / z, y / z


def laser_model(x_true, tl):
    x1 = x_true[0, 0]
    y1 = x_true[1, 0]
    theta1 = x_true[2, 0]
    # r = laser range, tl - theta laser
    r = 3.5
    # tl = 0
    theta2 = theta1 + tl
    x2 = x1 + r*math.cos(theta2)
    y2 = y1 + r*math.sin(theta2)

    pl1 = np.array([x1, y1])
    pl2 = np.array([x2, y2])
    r_error = 0.1**2 * np.random.randn(1)

    if intersect(P51, P52, pl1, pl2):
        laser_scan = get_intersect(P51, P52, pl1, pl2)
        r = np.linalg.norm(pl1-laser_scan)
    elif intersect(P11, P12, pl1, pl2):
        laser_scan = get_intersect(P11, P12, pl1, pl2) 
        r = np.linalg.norm(pl1-laser_scan)
    elif intersect(P21, P22, pl1, pl2):
        laser_scan = get_intersect(P21, P22, pl1, pl2) 
        r = np.linalg.norm(pl1-laser_scan)
    elif intersect(P31, P32, pl1, pl2):
        laser_scan = get_intersect(P31, P32, pl1, pl2) 
        r = np.linalg.norm(pl1-laser_scan)
    elif intersect(P41, P42, pl1, pl2):
        laser_scan = get_intersect(P41, P42, pl1, pl2) 
        r = np.linalg.norm(pl1-laser_scan)
   # elif intersect(P51, P52, pl1, pl2):
    #    laser_scan = get_intersect(P51, P52, pl1, pl2) 
     #   r = np.linalg.norm(pl1-laser_scan)
    else:
        laser_scan = float('inf'), float('inf')
        r = float('inf')

    r = r + r_error
    # laser_scan = laser_scan + (r_error*math.cos(tl), r_error*math.sin(tl))
    return laser_scan, r, r_error

"""
Codigo para parte de extract lines 


#funcoes para extract lines com dados do lazer
def pol2cart(theta, rho):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return(x, y)

def splitlines(xy, startidx, endidx, thersholds)
    return(alpha, r, idx)

def mergecolinear(xy, alpha, r, pointsidx, thersholds):
    return(alphaout, rout, pointsidxout)

#substituir theta e rho pelos dados do lazer
def extractlines(theta, rho, thersholds):
    #passa de coordenadas polares para cartesianas
    xy = np.zeros((1,0))
    xy = pol2cart(theta, rho)

    #faz a extracao das linhas
    alpha, r, pointsidx = splitlines(xy, 0, len(XY, 1), thersholds)

    #numero de segmentos de reta, caso seja mais do que um segmento, vereifica se sao colineares
    n= len(r)
    if n>1:
        alpha, r, pointidx = mergecolinear(xy, alpha, r, pointsidx, thersholds)
        n= len(r)
        #atualiza o numero de segmentos

    #definir coordenads dos endpoints e len dos segmentos
    segmends = np.zeros(n-1, 3)
    segmlen = np.zeros(n-1, 0)

    for l in range(0, n-1):
        segmends[l, :] =
        segmlen[l] = math.sqrt((segmends((l,0)) - segmends((l,2)))**2 + (segmends((l,1)) - segmends((l,3)))**2)

    #remover segmentos demasiados pequenos ???

    #definiçao de z, R
    z = np.zeros((len(alpha)-1, len(r)-1))
    z = ([[alpha],[r]])
    
    return z, r, segmends
    
"""

# Split and merge funçoes

class Thresholds:
    def __init__(self):
        self.seg_min_length = 0.01
        self.point_dist = 0.05
        self.min_point_seg = 5


def fitline(pontos):
    # centroid de pontos considerando que o centroid de
    # um numero finito de pontos pode ser obtido como
    # a media de cada coordenada

    lixo, len = pontos.shape
    # alpha = np.zeros((1,1))

    xc, yc = pontos.sum(axis=1) / len
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
        #print(z)
        zt[0], zt[1] = fitline(xy[:, startidx:(endidx + 1)])

        splitpos = findsplitpos(xy[:, startidx:(endidx + 1)], zt[0], zt[1], thresholds)
        zt[1] = np.matrix.item(zt[1])
        # Se nao for necessario fazer split, fazemos merge
        #print(zt[1])
        if splitpos == -1:
            z = zt
        else:  # Sem mais merges
            # alphaOut[j, 0] = z[0]
            alphaOut.append(z[0])
            #print(z)
            #print(z[1][0, 0])
            #list = np.matrix.tolist(z[1])
            #print(list)
            rOut.append(z[1])
            #print(rOut)
            # rOut[j, 0] = z[1]
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
    rOut = np.asmatrix(rOut)
    #print(rOut)


    return alphaOut, rOut, pointidxOut


def pol2cart(theta, rho):
    x = np.zeros((1, theta.shape[0]))
    y = np.zeros((1, theta.shape[0]))
    for i in range(0, theta.shape[0]):
        x[0, i] = rho[i, 0] * np.cos(theta[i, 0])
        y[0, i] = rho[i, 0] * np.sin(theta[i, 0])
    return x, y


def extractlines(theta, rho, thersholds):
    # passa de coordenadas polares para cartesianas

    x, y = pol2cart(theta, rho)

    xy = np.vstack((x, y))

    # xy = np.concatenate((x,y),axis=0)
    xy = np.asmatrix(xy)

    # print(xy)

    startidx = 0
    endidx = xy.shape[1] - 1  # x e y são vetores linha

    # faz a extracao das linhas
    alpha, r, pointsidx = splitlines(xy, startidx, endidx, thersholds)

    # numero de segmentos de reta, caso seja mais do que um segmento, vereifica se sao colineares
    n = r.shape[0]
    if n > 1:
        alpha, r, pointsidx = mergeColinear(xy, alpha, r, pointsidx, thersholds)
        #HA AQUI UM PROBLEMA NO R
        n = r.shape[0]
        # atualiza o numero de segmentos

    # definir coordenads dos endpoints e len dos segmentos
    segmends = np.zeros((n, 4))
    segmlen = np.zeros((n, 1))
    # for l in range(0, n):
    #    print(np.concatenate([np.transpose(xy[:, pointsidx[l, 0]]), np.transpose(xy[:, pointsidx[l, 1]])], axis = 1))
    pointsidx = np.asmatrix(pointsidx)
    for l in range(0, n):
        segmends[l, :] = np.concatenate([np.transpose(xy[:, pointsidx[l, 0]]), np.transpose(xy[:, pointsidx[l, 1]])],
                                        axis=1)
        # segmends[l, :] = [np.transpose(xy[:, pointsidx[l, 0]]), np.transpose(xy[:, pointsidx[l, 1]])]
        # for j in range(0:4):
        #    segmends[l, j] = [xy[j, pointsidx[l, 0]]]
        segmlen[l] = math.sqrt((segmends[l, 0] - segmends[l, 2]) ** 2 + (segmends[l, 1] - segmends[l, 3]) ** 2)

    segmlen = np.transpose(segmlen)
    # print(((pointsidx[:,1] - pointsidx[:,0]) >= thersholds.min_point_seg))
    # print((segmlen >= thersholds.seg_min_length))
    # print((segmlen >= thersholds.seg_min_length) & ((pointsidx[:,1] - pointsidx[:,0]) >= thersholds.min_point_seg))

    # remover segmentos demasiados pequenos
    # alterar thersholds para params.MIN_SEG_LENGTH e params.MIN_POINTS_PER_SEGMENT
    goodsegmidx = np.argwhere(
        np.transpose(segmlen >= thersholds.seg_min_length) & ((pointsidx[:, 1] - pointsidx[:, 0]) >= thersholds.min_point_seg))
    # print(goodsegmidx)
    # goodsegmix2 = goodsegmidx[0, 1]:goodsegmidx[(goodsegmidx.shape[0]), 1]
    # print(goodsegmix2)

    '''
    print('1a condicao')
    print(segmlen >= thersholds.seg_min_length)
    print('2a condicao')
    print((pointsidx[:, 1] - pointsidx[:, 0]) >= thersholds.min_point_seg)
    print('and')
    print(
        np.transpose(segmlen >= thersholds.seg_min_length) & ((pointsidx[:, 1] - pointsidx[:, 0]) >= thersholds.min_point_seg))

    print('goodsegmidx')
    print(goodsegmidx)
    '''
    pointsidx = pointsidx[goodsegmidx[:, 0], :]

    #print(pointsidx)

    alpha = np.asmatrix(alpha)
    alpha = alpha[goodsegmidx[:, 0], 0]
    #r = np.asmatrix(r)
    #print(r)
    r = r[goodsegmidx[:, 0], 0]
    # print(segmends)
    segmends = segmends[goodsegmidx[:, 0], :]
    segmlen = np.transpose(segmlen)
    segmlen = segmlen[goodsegmidx[:, 0], 0]

    #print(alpha)
    #print(r)
    # z = np.zeros((alpha.shape[0] - 1, r.shape[0] - 1))
    z = np.transpose(np.hstack((alpha, r))) #mudei para hstack


    R_seg = np.zeros((1, 1, len([len(alpha), 1]) - 1))

    return z, R_seg, segmends


if __name__ == '__main__':
    v = 0.1
    omega = 0.1
    u = np.array([[v * DT], [omega * DT]])

    time = 0.0
    i = 0

    # vetor de estado [x y theta]
    xTrue = np.zeros((3, 1))
    xPr = np.zeros((3, 1))
    xDR = xTrue
    xPred = np.zeros((3, 1))
    xEst = xTrue


    EEst = np.eye(3)

    # anteriores
    xPr_plot = xPr
    xDR_plot = xDR
    xPred_plot = xPred
    xEst_plot = xEst
    xTrue_plot = xTrue
    
    i = len(np.arange(-2.356194496154785, 2.0923497676849365, 0.05))
        # i += 1
    
    scan_m = np.zeros((2, i))


    thresholds = Thresholds()

    #print(seg_intersect(P11,P12,P21,P22))

    # hz = np.zeros((2, 1))
    while time <= 64:
        time += DT
        j = 0

        xTrue, xDR, ud = observation(xTrue, xDR, u)
        xEst, EEst = ekf_estimation(xEst, EEst, ud)

        # fazer historico de dados (para plot)
        xEst_plot = np.hstack((xEst_plot, xEst))
        xDR_plot = np.hstack((xDR_plot, xDR))
        xTrue_plot = np.hstack((xTrue_plot, xTrue))
        # scan_point = laser_model(xTrue)

        dist = np.zeros((i, 1))
        thetas = np.zeros((i, 1))
        # simulaçao
        plt.cla()

        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect('key_release_event',
                                     lambda event: [exit(0) if event.key == 'escape' else None])
        plot_map()

        plt.plot(xTrue_plot[0, :].flatten(),
                 xTrue_plot[1, :].flatten(), "-b")
        plt.plot(xDR_plot[0, :].flatten(),
                 xDR_plot[1, :].flatten(), "-k")
        plt.plot(xEst_plot[0, :].flatten(),
                 xEst_plot[1, :].flatten(), "-r")

        for tl in np.arange(-2.356194496154785, 2.0923497676849365, 0.05):
            scan_point, rang, rang_error = laser_model(xTrue, tl)
            """
            r_error = 0.1**2 * np.random.randn(1)
            scan_m[0,j] = scan_point[0] +  r_error*math.cos(tl)
            scan_m[1,j] = scan_point[1] +  r_error*math.sin(tl)
            """
            if scan_point != (float('inf'), float('inf')):       
                plt.scatter(scan_point[0] + rang_error*math.cos(tl), scan_point[1] + rang_error*math.sin(tl), 5, '#e10600', ",", zorder=100)
            scan_m[0, j] = rang
            scan_m[1, j] = tl
            # print(scan_m[:, j])
            j += 1
            
            # print(rang)
        f = 0

        for k in range(0, scan_m.shape[1]):
            dist[f] = scan_m[0, k]
            thetas[f] = scan_m[1, k]
            if scan_m[0, k] == float('inf'):
                dist = np.delete(dist, f)
                thetas = np.delete(thetas, f)
                f -= 1
            f += 1
        dist = np.transpose(np.asmatrix(dist))
        thetas = np.transpose(np.asmatrix(thetas))

        #print("dist")
        #print(dist)
        #print("thetas")
        #print(thetas)



        z, R, asase = extractlines(thetas, dist, thresholds)

        #print(z)
        for monkey in range (0, asase.shape[0]):
            asase = np.array(asase)
            point1 = [asase[monkey,0], asase[monkey,1]]
            point2 = [asase[monkey,2], asase[monkey,3]]
            x_values = [point1[0], point2[0]]
            y_values = [point1[1], point2[1]]
            plt.axis([-3.5, 3.5, -3.5, 3.5])
            plt.plot(x_values, y_values, '#e10600')
    
        #plot_covariance_ellipse(xEst, EEst)
        
        # plt.axis("equal")
        plt.axis([-3.5, 3.5, -3.5, 3.5])
        plt.grid(True)
        plt.pause(0.001)




       # plt.show()
