from cmath import pi
import sys

import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

def fitLinePolar(theta, rho, C_TR):
    lin_theta, N = theta.shape
    if lin_theta != 1 or rho.shape[0] != 1
        print('fitLinePolar only accepts column vectors')
        exit()
    if rho.shape[1] != N
        print('theta and rho must have matching size. But columns_theta == ' + theta.shape[1], end = '')
        print(", and columns_rho ==" + rho.shape[1])
        exit()
