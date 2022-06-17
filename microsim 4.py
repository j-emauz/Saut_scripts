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

Q_est = np.diag([
    0.1,  # variance of location on x-axis
    0.1,  # variance of location on y-axis
    np.deg2rad(2.0),  # variance of theta
]) ** 2  


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


def predict(x_est, E_est, u):
    # _predict step
    # x_est é o anterior e vai ser atualizado no final
    G_x = np.array([[1.0, 0, -u[0, 0] * math.sin(x_est[2, 0] + u[1, 0])],
                    [0, 1.0, u[0, 0] * math.cos(x_est[2, 0] + u[1, 0])],
                    [0, 0, 1.0]])

    b = np.array([[u[0, 0] * math.cos(x_est[2, 0] + u[1, 0])],
                  [u[0, 0] * math.sin(x_est[2, 0] + u[1, 0])],
                  [u[1, 0]]])

    E_est = G_x @ E_est @ G_x.T + Q_est
    x_est = x_est + b

    return x_est, E_est


def predict_motion(x_real, x_pred, u):
    b = np.array([[u[0, 0] * math.cos(x_real[2, 0] + u[1, 0])],
                  [u[0, 0] * math.sin(x_real[2, 0] + u[1, 0])],
                  [u[1, 0]]])
    x_real = x_real + b

    u_e = u + INPUT_NOISE @ np.random.randn(2, 1)
    b_e = np.array([[u_e[0, 0] * math.cos(x_pred[2, 0] + u_e[1, 0])],
                   [u_e[0, 0] * math.sin(x_pred[2, 0] + u_e[1, 0])],
                   [u_e[1, 0]]])
    x_pred = x_pred + b_e

    return x_real, x_pred, u_e

# Helper function para funcao intersect
def inter_help(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

# Retornar true se segmentos AB e CD intersetarem
def check_intersect(A, B, C, D):
    return inter_help(A, C, D) != inter_help(B, C, D) and inter_help(A, B, C) != inter_help(A, B, D)


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

    if check_intersect(P51, P52, pl1, pl2):
        laser_scan = get_intersect(P51, P52, pl1, pl2)
        r = np.linalg.norm(pl1 - laser_scan)
    elif check_intersect(P11, P12, pl1, pl2):
        laser_scan = get_intersect(P11, P12, pl1, pl2)
        r = np.linalg.norm(pl1 - laser_scan)

    elif check_intersect(P61, P62, pl1, pl2):
        laser_scan = get_intersect(P61, P62, pl1, pl2)
        r = np.linalg.norm(pl1 - laser_scan)
    elif check_intersect(P71, P72, pl1, pl2):
        laser_scan = get_intersect(P71, P72, pl1, pl2)
        r = np.linalg.norm(pl1 - laser_scan)
    elif check_intersect(P81, P82, pl1, pl2):
        laser_scan = get_intersect(P81, P82, pl1, pl2)
        r = np.linalg.norm(pl1 - laser_scan)
    elif check_intersect(P91, P92, pl1, pl2):
        laser_scan = get_intersect(P91, P92, pl1, pl2)
        r = np.linalg.norm(pl1 - laser_scan)

    elif check_intersect(P21, P22, pl1, pl2):
        laser_scan = get_intersect(P21, P22, pl1, pl2)
        r = np.linalg.norm(pl1 - laser_scan)
    elif check_intersect(P41, P42, pl1, pl2):
        laser_scan = get_intersect(P41, P42, pl1, pl2)
        r = np.linalg.norm(pl1 - laser_scan)
    elif check_intersect(P31, P32, pl1, pl2):
        laser_scan = get_intersect(P31, P32, pl1, pl2)
        r = np.linalg.norm(pl1 - laser_scan)

    else:
        laser_scan = float('inf'), float('inf')
        r = float('inf')

    r = r + r_error
    return laser_scan, r, r_error


# Funcoes split and merge
#POR COMENTARIOS THRESHOLDS
class Thresholds:
    def __init__(self):
        self.seg_min_length = 0.01
        self.point_dist = 0.05
        self.min_point_seg = 6


def line_regression(pontos):

    _, len = pontos.shape

    x_c, y_c = pontos.sum(axis=1) / len
    dx = (pontos[0, :] - x_c)
    dy = (pontos[1, :] - y_c)

    num = -2 * np.matrix.sum(np.multiply(dx, dy))
    den = np.matrix.sum(np.multiply(dy, dy) - np.multiply(dx, dx))
    alpha = math.atan2(num, den) / 2

    r = x_c * math.cos(alpha) + y_c * math.sin(alpha)

    if r < 0:
        alpha = alpha + math.pi
        if alpha > pi:
            alpha = alpha - 2 * math.pi
        r = -r

    return alpha, r


def dist2line(pontos, alpha, r):
    xcosa = pontos[0, :] * math.cos(alpha)
    ysina = pontos[1, :] * math.sin(alpha)
    d = xcosa + ysina - r
    return d


def split_position_id(d, thresholds):

    n_d = d.shape[1]
    d = abs(d)

    mask = d > thresholds.point_dist

    if not np.any(mask):
        split_position = -1
        return split_position

    split_position = np.argmax(d)

    if (split_position == 0):
        split_position = 1
    if (split_position == (n_d - 1)):
        split_position = n_d - 2
    return split_position


def find_split_position(pontos, alpha, r, thresholds):
    d = dist2line(pontos, alpha, r)
    split_position = split_position_id(d, thresholds)
    return split_position


def split_lines(pontos, i_id, f_id, thresholds):
    n_p = f_id - i_id + 1

    alpha, r = line_regression(pontos[:, i_id:(f_id + 1)])

    if n_p <= 2:
        ids = [i_id, f_id]
        return alpha, r, ids

    split_position = find_split_position(pontos[:, i_id:(f_id + 1)], alpha, r, thresholds)

    if (split_position != -1):
        alpha1, r1, idx1 = split_lines(pontos, i_id, split_position + i_id, thresholds)  # se calhar start ids-1
        alpha2, r2, idx2 = split_lines(pontos, split_position + i_id, f_id, thresholds)
        alpha = np.vstack((alpha1, alpha2))
        r = np.vstack((r1, r2))
        ids = np.vstack((idx1, idx2))
    else:
        ids = np.array([i_id, f_id])

    return alpha, r, ids


def merge_lines(pontos, alpha, r, p_ids, thresholds):
    z = [alpha[0, 0], r[0, 0]]
    i_id = p_ids[0, 0]
    last_id = p_ids[0, 1]

    n_lines = r.shape[0]
    z_t = [0, 0]

    r_out = []
    alpha_out = []
    p_ids_out = []

    j = 0

    for i in range(1, n_lines):
        f_id = p_ids[i, 1]

        z_t[0], z_t[1] = line_regression(pontos[:, i_id:(f_id + 1)])

        split_position = find_split_position(pontos[:, i_id:(f_id + 1)], z_t[0], z_t[1], thresholds)
        z_t[1] = np.matrix.item(z_t[1])

        if split_position == -1:
            z = z_t
        else:
            alpha_out.append(z[0])
            r_out.append(z[1])
            p_ids_out.extend([i_id, last_id])
            j = j + 1
            z = [alpha[i, 0], r[i, 0]]
            i_id = p_ids[i, 0]

        last_id = f_id

    # Adicionar o ultimo segmento
    alpha_out.append(z[0])
    r_out.append(z[1])
    p_ids_out.extend([i_id, last_id])

    p_ids_out = np.array(p_ids_out)
    p_ids_out = np.reshape(p_ids_out, (j + 1, 2))
    alpha_out = np.array(alpha_out)
    alpha_out = np.reshape(alpha_out, (j + 1, 1))
    r_out = np.array(r_out)
    r_out = np.reshape(r_out, (j + 1, 1))
    r_out = np.asmatrix(r_out)

    return alpha_out, r_out, p_ids_out


def pol2cart(theta, rho):
    x = np.zeros((1, theta.shape[0]))
    y = np.zeros((1, theta.shape[0]))
    for i in range(0, theta.shape[0]):
        x[0, i] = rho[i, 0] * np.cos(theta[i, 0])
        y[0, i] = rho[i, 0] * np.sin(theta[i, 0])
    return x, y


def split_merge(theta, rho, thersholds):

    x, y = pol2cart(theta, rho)

    pxy = np.vstack((x, y))
    pxy = np.asmatrix(pxy)

    i_id = 0
    f_id = pxy.shape[1] - 1  # x e y são vetores linha

    # faz a extracao das linhas
    alpha, r, p_ids = split_lines(pxy, i_id, f_id, thersholds)

    # numero de segmentos de reta, caso seja mais do que um segmento, verifica se sao colineares
    n = r.shape[0]
    if n > 1:
        alpha, r, p_ids = merge_lines(pxy, alpha, r, p_ids, thersholds)
        n = r.shape[0]
        # atualiza o numero de segmentos

    # definir coordenads dos endpoints e len dos segmentos
    seg_i_f = np.zeros((n, 4))
    seg_len = np.zeros((n, 1))

    p_ids = np.asmatrix(p_ids)

    if p_ids.shape[0]!=0:
        for l in range(0, n):
            seg_i_f[l, :] = np.concatenate([np.transpose(pxy[:, p_ids[l, 0]]), np.transpose(pxy[:, p_ids[l, 1]])],
                                            axis=1)
            seg_len[l] = math.sqrt((seg_i_f[l, 0] - seg_i_f[l, 2]) ** 2 + (seg_i_f[l, 1] - seg_i_f[l, 3]) ** 2)

    seg_len = np.transpose(seg_len)

    # remover segmentos demasiados pequenos
    correct_segs_ids = np.argwhere(
        np.transpose(seg_len >= thersholds.seg_min_length) & (
                    (p_ids[:, 1] - p_ids[:, 0]) >= thersholds.min_point_seg))

    '''
    print('1a condicao')
    print(seg_len >= thersholds.seg_min_length)
    print('2a condicao')
    print((p_ids[:, 1] - p_ids[:, 0]) >= thersholds.min_point_seg)
    print('and')
    print(
        np.transpose(seg_len >= thersholds.seg_min_length) & ((p_ids[:, 1] - p_ids[:, 0]) >= thersholds.min_point_seg))

    print('correct_segs_ids')
    print(correct_segs_ids)
    '''
    p_ids = p_ids[correct_segs_ids[:, 0], :]

    alpha = np.asmatrix(alpha)
    alpha = alpha[correct_segs_ids[:, 0], 0]

    r = r[correct_segs_ids[:, 0], 0]

    seg_i_f = seg_i_f[correct_segs_ids[:, 0], :]
    seg_len = np.transpose(seg_len)
    seg_len = seg_len[correct_segs_ids[:, 0], 0]

    z = np.transpose(np.hstack((alpha, r)))
    R_seg = np.zeros((2, 2, alpha.shape[0]))

    for c in range(0,alpha.shape[0]):
        for j in range(0,2):
            R_seg[j,j,c] = 0.5

    return z, R_seg, seg_i_f


def normalize_line(alpha, r):
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


def update_mat(x, m):
    h = np.array([[m[0] - x[2,0]], [m[1] - (x[0,0] * math.cos(m[0]) + x[1,0] * math.sin(m[0]))]])
    Hxmat = np.array([[0, 0, -1], [-math.cos(m[0]), -math.sin(m[0]), 0]])

    [h[0], h[1], isdistneg] = normalize_line(h[0], h[1])

    if isdistneg:
        Hxmat[1, :] = -Hxmat[1, :]

    return h, Hxmat


def matching(x, P, z, R_seg, M, g):
    #z: Linhas observadas
    n_measurs = z.shape[1]
    n_map = M.shape[1]

    d = np.zeros((n_measurs, n_map))
    v = np.zeros((2, n_measurs * n_map))
    H = np.zeros((2, 3, n_measurs * n_map ))

    v = np.asmatrix(v)


    for aux_nme in range(0, n_measurs):
        for aux_nmap in range(0, n_map):
            z_predict, H[:, :, aux_nmap + (aux_nme) * n_map] = update_mat(x, M[:, aux_nmap])
            v[:, aux_nmap + (aux_nme) * n_map] = z[:, aux_nme] - z_predict
            W = H[:, :, aux_nmap + (aux_nme) * n_map] @ P @ np.transpose(H[:, :, aux_nmap + (aux_nme) * n_map]) + R_seg[:, :, aux_nme]
            #Distancia Mahalanahobis
            d[aux_nme, aux_nmap] = np.transpose(v[:, aux_nmap + (aux_nme) * n_map]) * np.linalg.inv(W) * v[:, aux_nmap + (aux_nme) * n_map]


    min_mahal, map_id = (np.transpose(d)).min(0), (np.transpose(d)).argmin(0)
    measure_id = np.argwhere(min_mahal < g**2)
    map_id = map_id[np.transpose(measure_id)]
    seletor = (map_id + (np.transpose(measure_id))* n_map)
    seletorl =[]

    for f in range(0,seletor.shape[1]):
        seletorl.append(seletor.item(f))

    v = v[:, seletorl]
    H = H[:, :, seletorl]

    measure_id = np.transpose(measure_id)
    measure_idl = []
    for b in range(0, measure_id.shape[1]):
        measure_idl.append(measure_id.item(b))
    if seletorl == []:
        R_seg = R_seg[:, :, seletorl]
    else:
        R_seg = R_seg[:, :, measure_idl]

    return v, H, R_seg


def update(x_est, E_est, z, R_seg, mapa, g):

    if z.shape[1]==0:
        x_up = x_est
        E_up = E_est

        return x_up, E_up

    v, H, R_seg = matching(x_est, E_est, z, R_seg, mapa, g)

    #mudar formato de v, H e R para usar nas equacoes
    y = np.reshape(v, (v.shape[0]*v.shape[1],1), 'F')

    H = np.transpose(H, [0, 2, 1])
    Hreshape = np.reshape(H, [-1, 3], 'F')

    if R_seg.shape[2] == 0:
        R_seg1 = []
    else:
        R_seg1 = R_seg[:, :, 0]
        for bruh in range(1, R_seg.shape[2]):
            R_seg1 = scipy.linalg.block_diag(R_seg1, R_seg[:, :, bruh])

    S = Hreshape @ E_est @ np.transpose(Hreshape) + R_seg1
    K = E_est @ np.transpose(Hreshape) @ (np.linalg.inv(S))

    E_up = E_est - K @ S @ np.transpose(K)
    x_up = x_est + K @ y

    return x_up, E_up


def plot_covariance_ellipse(x_est, P_est):
    Pxy = P_est[0:2, 0:2]
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
    ex = [a * math.cos(ua) for ua in ta]
    ey = [b * math.sin(ua) for ua in ta]
    angle = math.atan2(eigvec[1, max], eigvec[0, max]) # angulo de rotaçao calculado com o valor proprio maior

    #matriz de rotação
    Rot = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])

    pe = Rot @ (np.array([ex, ey])) # pontos elipse apos rotaçao rotaçao da elipsoide
    px = np.array(pe[0, :] + x_est[0, 0]).flatten() # centrar 
    py = np.array(pe[1, :] + x_est[1, 0]).flatten() # centrar
    plt.plot(px, py, "--r")


def plots_x(x_real_plot, x_pred_plot, x_est_plot):
    plt.gcf().canvas.mpl_connect('key_release_event',
                                 lambda event: [exit(0) if event.key == 'escape' else None])
    plot_map()

    plt.plot(x_real_plot[0, :].flatten(),
             x_real_plot[1, :].flatten(), "-b")
    plt.plot(x_pred_plot[0, :].flatten(),
             x_pred_plot[1, :].flatten(), "-k")
    plt.plot(x_est_plot[0, :].flatten(),
             x_est_plot[1, :].flatten(), "-r")

    plt.axis("equal")
    plt.grid(True)
    plt.axis('equal')


if __name__ == '__main__':
    v = 0.1
    omega = 0.1
    time = 0.0
    i = 0

    # vetor de estado [x y theta]
    x_real = np.zeros((3, 1))
    x_real[0] = -1
    x_real[1] = -4
    x_real[2] = pi/2

    # x_pred = x_real
    x_pred = np.zeros((3,1))
    x_pred[0] = -1
    x_pred[1] = -4
    x_pred[2] = pi/2

    x_est = x_pred
    E_est = np.eye(3)

    x_pred_plot = x_pred
    x_est_plot = x_est
    x_real_plot = x_real
    t_traj = 0

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
        t_traj += 1
        if t_traj == 30:
            omega = -omega
            t_traj = -30

        u = np.array([[v * DT], [omega * DT]])
        time += DT
        j = 0
        dist = np.zeros((i, 1))
        thetas = np.zeros((i, 1))
        x_real, x_pred, u_e = predict_motion(x_real, x_pred, u)

        x_est, E_est = predict(x_est, E_est, u_e)
        for tl in np.arange(-2.356194496154785, 2.0923497676849365, 0.05):
            scan_point, rang, rang_error = laser_model(x_real, tl)

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

        z, R, seg_i_f = split_merge(thetas, dist, thresholds)

        x_est, E_est = update(x_est, E_est, z, R, mapa, g)

        x_est = np.asarray(x_est)
        # fazer historico de dados (para plot)
        x_est_plot = np.hstack((x_est_plot, x_est))
        x_pred_plot = np.hstack((x_pred_plot, x_pred))
        x_real_plot = np.hstack((x_real_plot, x_real))


        # simulaçao

        plt.gcf().canvas.mpl_connect('key_release_event',
                                     lambda event: [exit(0) if event.key == 'escape' else None])
        plot_map()

        plt.plot(x_real_plot[0, :].flatten(),
                 x_real_plot[1, :].flatten(), "-b")
        plt.plot(x_pred_plot[0, :].flatten(),
                 x_pred_plot[1, :].flatten(), "-k")
        plt.plot(x_est_plot[0, :].flatten(),
                 x_est_plot[1, :].flatten(), "-r")
                 


        """
        for mo in range(0, seg_i_f.shape[0]):
            seg_i_f = np.array(seg_i_f)
            point1 = [seg_i_f[mo, 0], seg_i_f[mo, 1]]
            point2 = [seg_i_f[mo, 2], seg_i_f[mo, 3]]
            x_values = [point1[0], point2[0]]
            y_values = [point1[1], point2[1]]
            plt.axis([-3.5, 3.5, -3.5, 3.5])
            plt.plot(x_values, y_values, '#03adfc')
        """
        plot_covariance_ellipse(x_est, E_est)


        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.001)
        # print(time)



        print(time)

    #plots_x(x_real_plot, x_pred_plot, x_est_plot)
    plt.show()
    # dif2 = x_real_plot-x_pred_plot
    dif1 = x_real_plot - x_est_plot

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
    
