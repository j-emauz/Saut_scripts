"""
@file hough_lines.py
@brief This program demonstrates line finding with the Hough transform
"""
import sys
import math
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pylab import *

lidar = np.linspace(-5, 5, 640)
def main(argv):

    default_file = 'monkeysmol2.pgm'
    filename = argv[0] if len(argv) > 0 else default_file
    # Loads an image
    src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_GRAYSCALE)
    # Check if image is loaded fine
    if src is None:
        print ('Error opening image!')
        print ('Usage: hough_lines.py [image_name -- default ' + default_file + '] \n')
        return -1


    dst = cv.Canny(src, 50, 300, None, 3)

    # Copy edges to the images that will display the results in BGR
    #cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    cdst = cv.cvtColor(src, cv.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)

    lines = cv.HoughLines(dst, 1, np.pi / 180, 50, None, 0, 0)

    figure = plt.figure(figsize=(12, 12))
    subplot = figure.add_subplot(1, 1, 1)
    subplot.set_facecolor((0, 0, 0))

    rho_graph1, rho_graph2 = np.zeros(100), np.zeros(100)
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv.line(cdst, pt1, pt2, (0,0,255), 3, cv.LINE_AA)

            theta_graph = np.linspace(-pi, pi, 100)

            #for j in range(0, 100):
            #    rho_graph1[j] = pt1[0]*np.cos(theta_graph[j]) + pt1[1]*np.sin(theta_graph[j])
            #    rho_graph2[j] = pt2[0]*np.cos(theta_graph[j]) + pt2[1]*np.sin(theta_graph[j])

            rho_graph1 = pt1[0]*np.cos(theta_graph) + pt1[1]*np.sin(theta_graph)
            rho_graph2 = pt2[0]*np.cos(theta_graph) + pt2[1]*np.sin(theta_graph)

            subplot.plot(theta_graph, rho_graph1, color="orange", alpha=0.05)
            subplot.plot(theta_graph, rho_graph2, color="orange", alpha=0.05)
            #subplot.plot([theta], [rho], marker='o', color="yellow")

    #subplot.invert_yaxis()
    #subplot.invert_xaxis()
    plt.show()

    #Parâmetros a mudar:
    #(ver: https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html)
    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 22, None, 20, 10)
    #linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)
    #linesP[0, 0, :] = [20, 400, 400, 20]

    #cenas = list(range(0, len(linesP)))
    #conjunto de linhas que selecionámos primeiramente:
    #cenas = [1, 2, 3, 11, 12, 13, 14, 15, 25, 33, 34, 37, 44]

    #linha que temos dúvidas: 2 -> alterar por: 16, 29 (descartei a 26, 49, 53)
    #adicionei ainda a 45 que é tipo lá do fundo
    cenas = [1, 2, 3, 11, 12, 13, 14, 15, 25, 33, 34, 37, 44, 16, 29, 45]

    #as que estou a tirar do conjunto total: 20, 51
    #cenas = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 52, 53]
    #cenas = list(range(0,len(linesP)))


    

    if linesP is not None:
        for i in range(0, len(cenas)):
        #for i in range(0, len(linesP)):#len(linesP)):  1,2,3,11,12,13,14,15,25,33,34,37,44 (6,7,8,9,16,17,18,19,20,21,22,23,24,26,27,28,29,30,31,32,35,36,38,39,41,42,43,45,46,47,48 descartar)  (4,5,10 não sabemos ->escolhemos 14)
            #l = linesP[i][0]    #linha 40,48 foi a que o João fez mal

            l = linesP[cenas[i]][0]
            cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)
    #print(cdstP)

    cv.imshow("Source", src)
    #cv.imshow("Source", dst)
    cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)


    linha = linesP[cenas,:,:]

    #Mudança de referencial da imagem para o referencial do mapa:
    for i in range(0, len(cenas)):
        linha[i, 0, 1]= -linha[i, 0, 1] + (640 - 456)
        linha[i, 0, 0]= linha[i, 0, 0] - 424
        linha[i, 0, 3] = -linha[i, 0, 3] + (640 - 456)
        linha[i, 0, 2] = linha[i, 0, 2] - 424


    #Obtenção da matriz de linhas do mapa em coordenadas polares:
    linha_polar=np.zeros((2,len(cenas)))

    for i in range(0, len(cenas)):
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
            isRNegated = 1
        else:
            isRNegated = 0

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