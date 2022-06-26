"""
Este programa vai detetar as linhas do mapa do corredor utilizando Hough Transform
e retorna a matriz com as coordenadas polares das respetivas retas em relação ao referencial
do mapa.
"""
import sys
import math
import cv2 as cv
import numpy as np


def main(argv):

    default_file = 'mapa_final.pgm'
    filename = argv[0] if len(argv) > 0 else default_file
    # Loads an image
    src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_GRAYSCALE)
    # Check if image is loaded fine
    if src is None:
        print ('Error opening image!')
        print ('Usage: hough_lines.py [image_name -- default ' + default_file + '] \n')
        return -1


    dst = cv.Canny(src, 50, 300, None, 3)
    #cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)

    cdst = cv.cvtColor(src, cv.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)


    #Parâmetros a mudar:
    #(ver: https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html)
    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 22, None, 20, 10)


    #índices das linhas selecionadas:
    line_idx = [1, 2, 3, 11, 12, 13, 14, 15, 25, 33, 34, 37, 44, 16, 29, 45]


    if linesP is not None:
        for i in range(0, len(line_idx)):

            l = linesP[line_idx[i]][0]
            cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)

    cv.imshow("Source", src)
    cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)


    linha = linesP[line_idx,:,:]

    #Mudança de referencial da imagem para o referencial do mapa:
    for i in range(0, len(line_idx)):
        linha[i, 0, 1] = -linha[i, 0, 1] + (640 - 456)
        linha[i, 0, 0] = linha[i, 0, 0] - 424
        linha[i, 0, 3] = -linha[i, 0, 3] + (640 - 456)
        linha[i, 0, 2] = linha[i, 0, 2] - 424


    #Obtenção da matriz de linhas do mapa em coordenadas polares:
    linha_polar=np.zeros((2,len(line_idx)))

    for i in range(0, len(line_idx)):
        # efetua regressao de reta em coordenadas polares
        x_c = (linha[i, 0, 0] + linha[i, 0, 2])/2
        y_c = (linha[i, 0, 1] + linha[i, 0, 3]) / 2
        lx = [0, 2]
        ly = [1, 3]
        dx = np.asmatrix(linha[i, 0, lx] - x_c)
        dy = np.asmatrix(linha[i, 0, ly] - y_c)

        num = -2 * np.matrix.sum(np.multiply(dx, dy))
        den = np.matrix.sum(np.multiply(dy, dy) - np.multiply(dx, dx))
        alpha = math.atan2(num, den) / 2

        r = x_c * math.cos(alpha) + y_c * math.sin(alpha)

        if r < 0:
            alpha = alpha + math.pi
            r = -r

        if alpha > math.pi:
            alpha = alpha - 2 * math.pi
        elif alpha < -math.pi:
            alpha = alpha + 2 * math.pi

        linha_polar[1, i] = r * 0.05  #in meters
        linha_polar[0, i] = alpha

    print(linha_polar)

    cv.waitKey()
    return 0


if __name__ == "__main__":
    main(sys.argv[1:])