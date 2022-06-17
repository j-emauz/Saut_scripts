import sys

from cmath import pi
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import scipy.linalg

"""
# linha vertical esquerda baixo
P11 = np.array([-2, -5])
P12 = np.array([-2, 14])
X1 = [P11[0], P12[0]]
Y1 = [P11[1], P12[1]]
"""
#COM ZONA TIPO ELEVADOR
# linha vertical esquerda baixo
P11 = np.array([-2, -5])
P12 = np.array([-2, 5])
X1 = [P11[0], P12[0]]
Y1 = [P11[1], P12[1]]

# linha horizontal baixo
P21 = np.array([-2, 5])
P22 = np.array([-6, 5])
X2 = [P21[0], P22[0]]
Y2 = [P21[1], P22[1]]
# linha horizontal cima
P31 = np.array([-2, 8])
P32 = np.array([-6, 8])
X3 = [P31[0], P32[0]]
Y3 = [P31[1], P32[1]]

# linha vertical esquerda cima
P41 = np.array([-2, 8])
P42 = np.array([-2, 14])
X4 = [P41[0], P42[0]]
Y4 = [P41[1], P42[1]]

# linha vertical direita
P51 = np.array([0, 12])
P52 = np.array([0, -3])
X5 = [P51[0], P52[0]]
Y5 = [P51[1], P52[1]]
#linha horizontal
P61 = np.array([-2, -5])
P62 = np.array([4, -5])
X6 = [P61[0], P62[0]]
Y6 = [P61[1], P62[1]]
#linha horizontal
P71 = np.array([0, -3])
P72 = np.array([3, -3])
X7 = [P71[0], P72[0]]
Y7 = [P71[1], P72[1]]

P81 = np.array([-2, 14])
P82 = np.array([2, 14])
X8 = [P81[0], P82[0]]
Y8 = [P81[1], P82[1]]

P91 = np.array([0, 12])
P92 = np.array([2, 12])
X9 = [P91[0], P92[0]]
Y9 = [P91[1], P92[1]]


INPUT_NOISE = np.diag([0.1, np.deg2rad(2.0)]) ** 2
SIM_TIME = 62.8
DT = 0.2

REst = np.diag([
    0.1,  # variance of location on x-axis
    0.1,  # variance of location on y-axis
    np.deg2rad(2.0),  # variance of theta
]) ** 2  # predict state covariance


def plot_map():
    plt.plot(X1, Y1, '-k')

    plt.plot(X2, Y2, '-k')
    plt.plot(X3, Y3, '-k')
    plt.plot(X4, Y4, '-k')

    plt.plot(X5, Y5, '-k')
    plt.plot(X6, Y6, '-k')
    plt.plot(X7, Y7, '-k')
    plt.plot(X8, Y8, '-k')
    plt.plot(X9, Y9, '-k')


def ekf_estimation(xEst, Eest, u):
    # Predict step
    # xEst é o anterior e vai ser atualizado no final
    G_x = np.array([[1.0, 0, -u[0, 0] * math.sin(xEst[2, 0] + u[1, 0])],
                    [0, 1.0, u[0, 0] * math.cos(xEst[2, 0] + u[1, 0])],
                    [0, 0, 1.0]])

    b = np.array([[u[0, 0] * math.cos(xEst[2, 0] + u[1, 0])],
                  [u[0, 0] * math.sin(xEst[2, 0] + u[1, 0])],
                  [u[1, 0]]])

    Eest = G_x @ Eest @ G_x.T + REst
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


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


# Return true if line segments AB and CD intersect
def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def get_intersect(p11, p12, p21, p22):
    """ 
    Retorna o ponto de intercecao de retas a passar por p12,p11 e p22,p21.
    p11: [x, y] ponto na primeira retas
    p12: [x, y] outro ponto da primeira reta
    p21: [x, y] ponto da segunda reta
    p22: [x, y] outro ponto da segunda reta
    """
    s = np.vstack([p11, p12, p21, p22])  # s de stack
    h = np.hstack((s, np.ones((4, 1))))  # homogeneo
    l1 = np.cross(h[0], h[1])  # obter primeira linha
    l2 = np.cross(h[2], h[3])  # obter segunda linha
    x, y, z = np.cross(l1, l2)  # ponto de interceçao
    if z == 0:  # se linhas sao paralelas
        return float('inf'), float('inf')
    return x / z, y / z


def laser_model(x_true, tl):
    x1 = x_true[0, 0]
    y1 = x_true[1, 0]
    theta1 = x_true[2, 0]
    # r = laser range, tl - theta laser
    r = 3.5

    theta2 = theta1 + tl
    x2 = x1 + r * math.cos(theta2)
    y2 = y1 + r * math.sin(theta2)

    pl1 = np.array([x1, y1])
    pl2 = np.array([x2, y2])
    r_error = 0.1 ** 2 * np.random.randn(1)

    if intersect(P51, P52, pl1, pl2):
        laser_scan = get_intersect(P51, P52, pl1, pl2)
        r = np.linalg.norm(pl1 - laser_scan)
    elif intersect(P11, P12, pl1, pl2):
        laser_scan = get_intersect(P11, P12, pl1, pl2)
        r = np.linalg.norm(pl1 - laser_scan)

    elif intersect(P61, P62, pl1, pl2):
        laser_scan = get_intersect(P61, P62, pl1, pl2)
        r = np.linalg.norm(pl1 - laser_scan)
    elif intersect(P71, P72, pl1, pl2):
        laser_scan = get_intersect(P71, P72, pl1, pl2)
        r = np.linalg.norm(pl1 - laser_scan)
    elif intersect(P81, P82, pl1, pl2):
        laser_scan = get_intersect(P81, P82, pl1, pl2)
        r = np.linalg.norm(pl1 - laser_scan)
    elif intersect(P91, P92, pl1, pl2):
        laser_scan = get_intersect(P91, P92, pl1, pl2)
        r = np.linalg.norm(pl1 - laser_scan)

    elif intersect(P21, P22, pl1, pl2):
        laser_scan = get_intersect(P21, P22, pl1, pl2)
        r = np.linalg.norm(pl1 - laser_scan)
    elif intersect(P41, P42, pl1, pl2):
        laser_scan = get_intersect(P41, P42, pl1, pl2)
        r = np.linalg.norm(pl1 - laser_scan)
    elif intersect(P31, P32, pl1, pl2):
        laser_scan = get_intersect(P31, P32, pl1, pl2)
        r = np.linalg.norm(pl1 - laser_scan)

    else:
        laser_scan = float('inf'), float('inf')
        r = float('inf')

    r = r + r_error
    return laser_scan, r, r_error


# Split and merge funçoes

class Thresholds:
    def __init__(self):
        self.seg_min_length = 0.01
        self.point_dist = 0.05
        self.min_point_seg = 6


def fitline(pontos):

    _, len = pontos.shape

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

    N = d.shape[1]
    d = abs(d)

    mask = d > thresholds.point_dist

    if not np.any(mask):
        splitpos = -1
        return splitpos

    splitpos = np.argmax(d)

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

    rOut = []
    alphaOut = []
    pointidxOut = []

    j = 0

    for i in range(1, N):
        endidx = pointidx[i, 1]

        zt[0], zt[1] = fitline(xy[:, startidx:(endidx + 1)])

        splitpos = findsplitpos(xy[:, startidx:(endidx + 1)], zt[0], zt[1], thresholds)
        zt[1] = np.matrix.item(zt[1])

        if splitpos == -1:
            z = zt
        else:
            alphaOut.append(z[0])
            rOut.append(z[1])
            pointidxOut.extend([startidx, lastendidx])
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

    return alphaOut, rOut, pointidxOut


def pol2cart(theta, rho):
    x = np.zeros((1, theta.shape[0]))
    y = np.zeros((1, theta.shape[0]))
    for i in range(0, theta.shape[0]):
        x[0, i] = rho[i, 0] * np.cos(theta[i, 0])
        y[0, i] = rho[i, 0] * np.sin(theta[i, 0])
    return x, y


def extractlines(theta, rho, thersholds):

    x, y = pol2cart(theta, rho)

    xy = np.vstack((x, y))
    xy = np.asmatrix(xy)

    startidx = 0
    endidx = xy.shape[1] - 1  # x e y são vetores linha

    # faz a extracao das linhas
    alpha, r, pointsidx = splitlines(xy, startidx, endidx, thersholds)

    # numero de segmentos de reta, caso seja mais do que um segmento, verifica se sao colineares
    n = r.shape[0]
    if n > 1:
        alpha, r, pointsidx = mergeColinear(xy, alpha, r, pointsidx, thersholds)
        n = r.shape[0]
        # atualiza o numero de segmentos

    # definir coordenads dos endpoints e len dos segmentos
    segmends = np.zeros((n, 4))
    segmlen = np.zeros((n, 1))

    pointsidx = np.asmatrix(pointsidx)

    if pointsidx.shape[0]!=0:
        for l in range(0, n):
            segmends[l, :] = np.concatenate([np.transpose(xy[:, pointsidx[l, 0]]), np.transpose(xy[:, pointsidx[l, 1]])],
                                            axis=1)
            segmlen[l] = math.sqrt((segmends[l, 0] - segmends[l, 2]) ** 2 + (segmends[l, 1] - segmends[l, 3]) ** 2)

    segmlen = np.transpose(segmlen)

    # remover segmentos demasiados pequenos
    goodsegmidx = np.argwhere(
        np.transpose(segmlen >= thersholds.seg_min_length) & (
                    (pointsidx[:, 1] - pointsidx[:, 0]) >= thersholds.min_point_seg))


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

    alpha = np.asmatrix(alpha)
    alpha = alpha[goodsegmidx[:, 0], 0]

    r = r[goodsegmidx[:, 0], 0]

    segmends = segmends[goodsegmidx[:, 0], :]
    segmlen = np.transpose(segmlen)
    segmlen = segmlen[goodsegmidx[:, 0], 0]

    z = np.transpose(np.hstack((alpha, r)))
    R_seg = np.zeros((2, 2, alpha.shape[0]))

    for c in range(0,alpha.shape[0]):
        for j in range(0,2):
            R_seg[j,j,c] = 0.5

    return z, R_seg, segmends


def normalizelineparameters(alpha, r):
    if r < 0:
        alpha = alpha + pi
        r = -r
        isRNegated = 1
    else:
        isRNegated = 0

    if alpha > math.pi:
        alpha = alpha - 2 * math.pi
    elif alpha < -math.pi:
        alpha = alpha + 2 * math.pi

    return alpha, r, isRNegated


def updatemat(x, m):
    h = np.array([[m[0] - x[2,0]], [m[1] - (x[0,0] * math.cos(m[0]) + x[1,0] * math.sin(m[0]))]])
    Hxmat = np.array([[0, 0, -1], [-math.cos(m[0]), -math.sin(m[0]), 0]])

    [h[0], h[1], isdistneg] = normalizelineparameters(h[0], h[1])

    if isdistneg:
        Hxmat[1, :] = -Hxmat[1, :]

    return h, Hxmat


def matching(x, P, Z, R_seg, M, g):
    #Z: Linhas observadas
    n_measurs = Z.shape[1]
    n_map = M.shape[1]

    d = np.zeros((n_measurs, n_map))
    v = np.zeros((2, n_measurs * n_map))
    H = np.zeros((2, 3, n_measurs * n_map ))

    v = np.asmatrix(v)


    for aux_nme in range(0, n_measurs):
        for aux_nmap in range(0, n_map):
            Z_predict, H[:, :, aux_nmap + (aux_nme) * n_map] = updatemat(x, M[:, aux_nmap])
            v[:, aux_nmap + (aux_nme) * n_map] = Z[:, aux_nme] - Z_predict
            W = H[:, :, aux_nmap + (aux_nme) * n_map] @ P @ np.transpose(H[:, :, aux_nmap + (aux_nme) * n_map]) + R_seg[:, :, aux_nme]
            #Distancia Mahalanahobis
            d[aux_nme, aux_nmap] = np.transpose(v[:, aux_nmap + (aux_nme) * n_map]) * np.linalg.inv(W) * v[:, aux_nmap + (aux_nme) * n_map]


    minima, mapidx = (np.transpose(d)).min(0), (np.transpose(d)).argmin(0)
    measursidx = np.argwhere(minima < g**2)
    mapidx = mapidx[np.transpose(measursidx)]
    seletor = (mapidx + (np.transpose(measursidx))* n_map)
    seletorl =[]

    for f in range(0,seletor.shape[1]):
        seletorl.append(seletor.item(f))

    v = v[:, seletorl]
    H = H[:, :, seletorl]

    measursidx = np.transpose(measursidx)
    measuridxl = []
    for b in range(0, measursidx.shape[1]):
        measuridxl.append(measursidx.item(b))
    if seletorl == []:
        R_seg = R_seg[:, :, seletorl]
    else:
        R_seg = R_seg[:, :, measuridxl]

    return v, H, R_seg


def step_update(x_pred, E_pred,  Z, R_seg, mapa, g):

    if Z.shape[1]==0:
        x_up = x_pred
        E_up = E_pred

        return  x_up, E_up

    v, H, R_seg = matching(x_pred, E_pred, Z, R_seg, mapa, g)

    #mudar formato de v, H e R para usar nas equacoes
    y = np.reshape(v, (v.shape[0]*v.shape[1],1), 'F')

    Hreshape = np.zeros((H.shape[0] * H.shape[2], 3))
    cenoura = 0
    for batata in range(0, H.shape[2]):
        Hreshape[cenoura, :] = H[0, :, batata]
        Hreshape[cenoura + 1, :] = H[1, :, batata]
        cenoura = cenoura + 2

    if R_seg.shape[2] == 0:
        R_seg1 = []
    else:
        R_seg1 = R_seg[:, :, 0]
        for bruh in range(1, R_seg.shape[2]):
            R_seg1 = scipy.linalg.block_diag(R_seg1, R_seg[:, :, bruh])

    S = Hreshape @ E_pred @ np.transpose(Hreshape) + R_seg1
    K = E_pred @ np.transpose(Hreshape) @ (np.linalg.inv(S))

    E_up = E_pred - K @ S @ np.transpose(K)
    x_up = x_pred + K @ y

    return x_up, E_up


def plot_covariance_ellipse(xEst, PEst):
    Pxy = PEst[0:2, 0:2]
    eigval, eigvec = np.linalg.eig(Pxy)

    if eigval[0] >= eigval[1]:
        max = 0
        min = 1
    else:
        max = 1
        min = 0

    ta = np.arange(0, 2 * pi + 0.1, 0.1) #todos os angulos
    a = math.sqrt(eigval[max])
    b = math.sqrt(eigval[min])
    x = [a * math.cos(ua) for ua in ta]
    y = [b * math.sin(ua) for ua in ta]
    angle = math.atan2(eigvec[1, max], eigvec[0, max]) # angulo de rotaçao calculado com o valor proprio maior

    #matriz de rotação
    Rot = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])

    fx = Rot @ (np.array([x, y]))
    px = np.array(fx[0, :] + xEst[0, 0]).flatten()
    py = np.array(fx[1, :] + xEst[1, 0]).flatten()
    plt.plot(px, py, "--r")


def plot_function(xTrue_plot, xDR_plot, xEst_plot):
    plt.gcf().canvas.mpl_connect('key_release_event',
                                 lambda event: [exit(0) if event.key == 'escape' else None])
    plot_map()

    plt.plot(xTrue_plot[0, :].flatten(),
             xTrue_plot[1, :].flatten(), "-b")
    plt.plot(xDR_plot[0, :].flatten(),
             xDR_plot[1, :].flatten(), "-k")
    plt.plot(xEst_plot[0, :].flatten(),
             xEst_plot[1, :].flatten(), "-r")

    plt.axis("equal")
    plt.grid(True)
    plt.axis('equal')


if __name__ == '__main__':
    v = 0.1
    omega = 0.1
    time = 0.0
    i = 0

    # vetor de estado [x y theta]
    xTrue = np.zeros((3, 1))
    xTrue[0] = -1
    xTrue[1] = -4
    xTrue[2] = pi/2

    # xDR = xTrue
    xDR = np.zeros((3,1))
    xDR[0] = -2
    xDR[1] = -4
    xDR[2] = pi/2

    xEst = xDR
    EEst = np.eye(3)

    xDR_plot = xDR
    xEst_plot = xEst
    xTrue_plot = xTrue
    tempao = 0

    i = len(np.arange(-2.356194496154785, 2.0923497676849365, 0.05))

    scan_m = np.zeros((2, i))
    thresholds = Thresholds()
    g = 0.5 #Threshold do matching

    #Mapa corredor comprido
    #mapa = np.array([[-pi / 2, -pi / 2, pi, pi / 2,  pi / 2, pi / 2], [5, 3, 2, 0,  12, 14]])

    #Mapa corredor com parte do elevador
    mapa = np.array([[-pi/2, -pi/2, pi, pi/2, pi/2, pi/2, pi/2, pi/2], [5, 3, 2, 0, 5, 8, 12, 14]])

    while time <= 180:
        plt.cla()
        tempao += 1
        if tempao == 30:
            omega = -omega
            tempao = -30

        u = np.array([[v * DT], [omega * DT]])
        time += DT
        j = 0
        dist = np.zeros((i, 1))
        thetas = np.zeros((i, 1))
        xTrue, xDR, ud = observation(xTrue, xDR, u)

        xEst, EEst = ekf_estimation(xEst, EEst, ud)
        for tl in np.arange(-2.356194496154785, 2.0923497676849365, 0.05):
            scan_point, rang, rang_error = laser_model(xTrue, tl)
            """
            r_error = 0.1**2 * np.random.randn(1)
            scan_m[0,j] = scan_point[0] +  r_error*math.cos(tl)
            scan_m[1,j] = scan_point[1] +  r_error*math.sin(tl)
            """
            if scan_point != (float('inf'), float('inf')):
                plt.scatter(scan_point[0] + rang_error * math.cos(tl), scan_point[1] + rang_error * math.sin(tl), 5,
                            '#e10600', ",", zorder=100)
            scan_m[0, j] = rang
            scan_m[1, j] = tl
            j += 1

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

        z, Q, segends = extractlines(thetas, dist, thresholds)

        xEst, EEst = step_update(xEst, EEst, z, Q, mapa, g)

        xEst = np.asarray(xEst)
        # fazer historico de dados (para plot)
        xEst_plot = np.hstack((xEst_plot, xEst))
        xDR_plot = np.hstack((xDR_plot, xDR))
        xTrue_plot = np.hstack((xTrue_plot, xTrue))


        # simulaçao

        plt.gcf().canvas.mpl_connect('key_release_event',
                                     lambda event: [exit(0) if event.key == 'escape' else None])
        plot_map()

        plt.plot(xTrue_plot[0, :].flatten(),
                 xTrue_plot[1, :].flatten(), "-b")
        plt.plot(xDR_plot[0, :].flatten(),
                 xDR_plot[1, :].flatten(), "-k")
        plt.plot(xEst_plot[0, :].flatten(),
                 xEst_plot[1, :].flatten(), "-r")
                 


        """
        for mo in range(0, segends.shape[0]):
            segends = np.array(segends)
            point1 = [segends[mo, 0], segends[mo, 1]]
            point2 = [segends[mo, 2], segends[mo, 3]]
            x_values = [point1[0], point2[0]]
            y_values = [point1[1], point2[1]]
            plt.axis([-3.5, 3.5, -3.5, 3.5])
            plt.plot(x_values, y_values, '#03adfc')
        """
        plot_covariance_ellipse(xEst, EEst)


        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.001)
        # print(time)



        print(time)

    #plot_function(xTrue_plot, xDR_plot, xEst_plot)
    plt.show()
    # dif2 = xTrue_plot-xDR_plot
    dif1 = xTrue_plot - xEst_plot

    max1x = np.amax(dif1[0, :])
    min1x = np.amin(dif1[0, :])
    
    max1y = np.amax(dif1[1, :])
    min1y = np.amin(dif1[1, :])
    
    max1t = np.amax(dif1[2, :])
    min1t = np.amin(dif1[2, :])
    
    
    print('max1x = ', max1x)
    print('min1x = ', min1x)
    
    print('max1y = ', max1y)  
    print('min1y = ', min1y)
    
    print('max1t = ', max1t)
    print('min1t = ', min1t)
    
