from cmath import pi
import sys

import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

def fitLinePolar(theta, rho, C_TR): #usectr = 0):
    lin_theta, N = theta.shape
    if lin_theta != 1 or rho.shape[0] != 1:
        print('fitLinePolar only accepts column vectors')
        exit()
    if rho.shape[1] != N:
        print('theta and rho must have matching size. But columns_theta == ' + theta.shape[1], end = '')
        print(", and columns_rho ==" + rho.shape[1])
        exit()

    rhoSquare = np.multiply(rho,rho)
    cs = math.cos(theta)
    cs2 = math.cos(np.multiply(2,theta))
    sn = math.sin(theta)
    sn2 = math.sin(np.multiply(2,theta))

    thetaTemp = np.multiply(np.transpose(theta),np.ones((1,N),Float))
    thetaDyadSum = thetaTemp + np.transpose(thetaTemp)
    cosThetaDyadSum = math.cos(thetaDyadSum)

    rhoDyad = np.multiply(np.transpose(rho),rho)
    csIJ = np.matrix.sum(np.matrix.sum(np.multiply(rhoDyad,cosThetaDyadSum)))

    #if C_TR
    sinThetaDyadSum = math.sin(thetaDyadSum)
    grad_thetaCsIJ = -np.matrix.sum()