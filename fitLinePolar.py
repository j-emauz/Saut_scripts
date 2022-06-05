from cmath import pi
import sys

import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv


def normalizeLineParameters(alpha,r):
    if r<0:
        alpha=alpha+pi
        r=-r
        isRNegated = 1
    else:
        isRNegated = 0

    if alpha > math.pi:
        alpha = alpha -2*math.pi
    elif alpha < -math.pi:
        alpha = alpha + 2*math.pi

    return alpha, r, isRNegated


def fitLinePolar(theta, rho, C_TR): #usectr = 0):
    N = theta.shape[1]
    if theta.shape[0] != 1 or rho.shape[0] != 1:
        print('fitLinePolar only accepts column vectors')
        exit()
    if rho.shape[1] != N:
        print('theta and rho must have matching size. But columns_theta == ' + theta.shape[1], end = '')
        print(", and columns_rho ==" + rho.shape[1])
        exit()

    rhoSquare = np.multiply(rho, rho)
    cs = math.cos(theta)
    cs2 = math.cos(2*theta)
    sn = math.sin(theta)
    sn2 = math.sin(2*theta)

    #thetaTemp = np.multiply(np.transpose(theta), np.ones((1, N), dtype=float))
    thetaTemp = np.multiply(np.transpose(theta), np.ones((1, N)))
    thetaDyadSum = thetaTemp + np.transpose(thetaTemp)
    cosThetaDyadSum = math.cos(thetaDyadSum)

    rhoDyad = np.transpose(rho) @ rho
    csIJ = np.matrix.sum(np.matrix.sum(np.multiply(rhoDyad, cosThetaDyadSum)))

    ######if C_TR
    sinThetaDyadSum = math.sin(thetaDyadSum)
    mult = np.multiply(rhoDyad, sinThetaDyadSum)
    grad_thetaCsIJ = -np.matrix.sum(mult, axis=0)-np.transpose(np.matrix.sum(mult, axis=1))
    grad_rhoCsIJ = 2 * rho @ cosThetaDyadSum
    ######

    y=rhoSquare @ np.transpose(sn2) -2/N * rho @ np.transpose(cs) @ rho @ np.transpose(sn)
    x=rhoSquare @ np.transpose(cs2) - csIJ /N

    alpha = 0.5 * (math.atan2(y, x)+math.pi)

    num2term=np.multiply(math.cos(theta-np.ones((len(theta),), dtype=float)), alpha)
    numR=np.multiply(rho, np.transpose(num2term))
    r= rho @ np.transpose(math.cos(theta - np.ones((len(theta),), dtype=float) @ alpha)) / N

    alphaOrg = alpha
    alpha, r, isRNegated = normalizeLineParameters(alpha, r)

    #####if C_TR
    rhosnT=rho @ np.transpose(sn)
    #AQUI ELE DISSE QUE ERA FLOATS PORTANTO NÃO DAVA PARA POR O @, SÃO FLOATS??
    grad_rhoY = 2 * np.multiply(sn2, rho)-2/N * (cs * rhosnT + sn * rhosnT)
    grad_rhoX = 2 * np.multiply(cs2, rho) - 1/N * grad_rhoCsIJ
    grad_thetaY=np.multiply(rhoSquare, (-2*sn2))-2/N*(np.multiply(rho, -sn)) @ (rho @ np.transpose(sn)) + np.multiply(rho, cs) @ (rho @ np.transpose(cs))
    grad_thetaX = np.multiply(rhoSquare, (-2*sn2)) - 1/N * grad_thetaCsIJ

    if x != 0:
        gradAlpha = 0.5/((y/x)**2 + 1) * ([[grad_thetaY], [grad_rhoY]] / x - np.multiply(y, x**(-2)) * [[grad_thetaX], [grad_rhoX]])
    else:
        gradAlpha = 0.5 * (-1/y) * [[grad_thetaX], [grad_rhoX]]

    grad_rhoR = (math.cos(theta - np.ones((len(theta),), dtype=float) * alphaOrg)+rho @ np.transpose(math.sin(theta - np.ones((theta.shape[0], theta.shape[1]), dtype=float) @ alphaOrg)) @ gradAlpha[1, N:(2*N-1)])/N
    term1=-math.sin(theta - np.ones((theta.shape[0], theta.shape[1]), dtype=float) @ alphaOrg)
    term2= np.multiply(rho, term1) @ (np.identity(N) - np.transpose(np.ones((theta.shape[0], theta.shape[1]), dtype=float)) @ gradAlpha[0, 1:N-1])
    grad_thetaR = term2 / N

    gradR = [[grad_thetaR], [grad_rhoR]]

    if isRNegated == 1:
        gradR=-gradR

    F_TR = [[gradAlpha], [gradR]]

    C_AR = F_TR @ C_TR @ np.transpose(F_TR)

    return alpha, r, C_AR
#??????????????????????????????????????????????
"""
def fitLinePolarNum(X):
   N=X.shape[0]
   alpha, r = fitLinePolar(, )
   return f
"""
