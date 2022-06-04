import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model, datasets
from skimage.measure import LineModelND, ransac
import pandas as pd
import math

df = pd.read_csv('scanData.txt',delimiter=',')
angle = df.values[:,0]
distance = df.values[:,1]
cartesian = [(r*math.cos(phi*math.pi/180), r*math.sin(phi*math.pi/180)) for r, phi in zip(distance, angle)]
x, y = map(list, zip(*cartesian))

# coverting this into 2d array
x_data =  np.array(x)
y_data =  np.array(y)

def plot_ransac(segment_data_x, segment_data_y):
    data = np.column_stack([segment_data_x, segment_data_y])

    # fit line using all data
    model = LineModelND()
    model.estimate(data)

    # robustly fit line only using inlier data with RANSAC algorithm
    model_robust, inliers = ransac(data, LineModelND, min_samples=2,
                                   residual_threshold=5, max_trials=1000)
    outliers = inliers == False

    # generate coordinates of estimated models
    line_x = np.array([segment_data_x.min(), segment_data_x.max()])
    line_y = model.predict_y(line_x)
    line_y_robust = model_robust.predict_y(line_x)
    k = (line_y_robust[1] - line_y_robust[0])/(line_x[1]- line_x[0])
    m = line_y_robust[0] - k*line_x[0]
    x0 = (segment_data_y.min() - m)/k
    x1 = (segment_data_y.max() - m)/k
    line_x_y = np.array([x0, x1])
    line_y_robust_y = model_robust.predict_y(line_x_y)
    if (distance(line_x[0], line_y_robust[0], line_x[1], line_y_robust[1]) <
    distance(line_x_y[0], line_y_robust_y[0], line_x_y[1], line_y_robust_y[1])):
        plt.plot(line_x, line_y_robust, '-b', label='Robust line model')
    else:
        plt.plot(line_x_y, line_y_robust_y, '-b', label='Robust line model')


x_segments = []
y_segments = []

def distance(x1,y1,x2,y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

start = 0
distances = []
for i in range(len(x_data)-1):
    distance_to_point = distance(x_data[i], y_data[i], x_data[i+1], y_data[i+1])
    distances.append(distance_to_point)
    if distance_to_point > 200:
        if i-start>10:
            x_segments.append(x_data[start:i])
            y_segments.append(y_data[start:i])
        start = i+1
    if i == len(x_data)-2:
        if i-start>10:
            x_segments.append(x_data[start:i])
            y_segments.append(y_data[start:i])

plt.plot(x_data, y_data, '.', color = 'grey')
for x_seg, y_seg in zip(x_segments, y_segments):
    plt.plot(x_seg, y_seg,'.', markersize = 10)
    plot_ransac(x_seg, y_seg)
    print('Line is:', distance(x_seg[0], y_seg[0],x_seg[1], y_seg[1]), 'units long')

plt.axis('equal')
plt.show()