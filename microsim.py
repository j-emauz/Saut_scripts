import sys

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
SIM_TIME = 64
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

    
    #print(seg_intersect(P11,P12,P21,P22))

    # hz = np.zeros((2, 1))
    while time <= SIM_TIME:
        time += DT
        j = 0

        xTrue, xDR, ud = observation(xTrue, xDR, u)
        xEst, EEst = ekf_estimation(xEst, EEst, ud)

        # fazer historico de dados (para plot)
        xEst_plot = np.hstack((xEst_plot, xEst))
        xDR_plot = np.hstack((xDR_plot, xDR))
        xTrue_plot = np.hstack((xTrue_plot, xTrue))
        # scan_point = laser_model(xTrue)
        
        
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
        
    
        #plot_covariance_ellipse(xEst, EEst)
        
        # plt.axis("equal")
        plt.axis([-3.5, 3.5, -3.5, 3.5])
        plt.grid(True)
        plt.pause(0.001)

       

