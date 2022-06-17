import sys

from cmath import pi
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import scipy.linalg

H = np.random.randint(10, size=(2, 3, 8))

print('H antes de alterar')
for i in range(0,H.shape[2]):
    print(H[:,:,i])

H = np.transpose(H, [0,2,1]) #permutar 2ª e 3ª dimensoes entre si

print('H depois do transpose/permute')
for i in range(0,H.shape[2]):
    print(H[:,:,i])

H = np.reshape(H, [-1,3], 'F')
print('H final')
print(H)

