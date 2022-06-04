import math
import numpy as np

def updatemat(x, m):
    h = np.array(([[m[0] - x[2]], [m[1]-(x[0]*math.cos(m[0] + x[1]*math.sin(m[0])))]]))
    Hxmat =  np.array(([[0, 0, -1], [-math.cos[m[0]], -math.sin[m[0]], 0]]))

    [h[0], h[1], isdistneg] = normalizelineparameters(h[0], h[1])

    if isdistneg:
        Hxmat[2, :] = -Hxmat[2, :]

    return h, Hxmat