"""
@file hough_lines.py
@brief This program demonstrates line finding with the Hough transform
"""
import sys
import math
import cv2 as cv
import numpy as np

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

    lines = cv.HoughLines(dst, 1, np.pi / 180, 40, None, 0, 0)

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

    #Parâmetros a mudar:
    #(ver: https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html)
    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 22, None, 20, 10)
    #linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)

    #linesP[0, 0, :] = [0, 0, 500, 500]
    cenas = [1, 2, 3, 11, 12, 13, 14, 15, 25, 33, 34, 37, 44]
    print(len(linesP))
    if linesP is not None:
        for i in range(0, len(cenas)):
        #for i in range(0, len(linesP)):#len(linesP)):  1,2,3,11,12,13,14,15,25,33,34,37,44 (6,7,8,9,16,17,18,19,20,21,22,23,24,26,27,28,29,30,31,32,35,36,38,39,41,42,43,45,46,47,48 descartar)  (4,5,10 não sabemos ->escolhemos 14)
            #l = linesP[i][0]    #linha 40,48 foi a que o João fez mal
            l = linesP[cenas[i]][0]
            cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)
    #print(cdstP)
    cv.imshow("Source", src)
    #cv.imshow("Source", dst)
    #cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)

    cv.waitKey()
    return 0

if __name__ == "__main__":
    main(sys.argv[1:])