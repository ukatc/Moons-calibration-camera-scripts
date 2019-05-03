"""
Determine the spread of errors in a pupil alignment configuration produced by
find_best_pupil_config.py by producing a heatmap where brighter pixels indicate higher deviation.

Based on a combination of find_best_pupil_config.py and the camera-calibration calibration repo's
example script analyse_precise_calibration.py
"""
import camera_calibration as calib
import cv2 as cv
import numpy as np
from pprint import pprint
import math

rows = 11
cols = 8

lengths_mm = [45.65, 91.13, 136.58, 64.30, 128.78]
lengths_edges = [1, 2, 3, math.sqrt(2), math.sqrt(2) * 2]

square_edge_mm = sum(
    lengths_mm[i] / lengths_edges[i] for i in range(len(lengths_mm))
) / len(lengths_mm)

grid_width = square_edge_mm * (cols - 1)
grid_height = square_edge_mm * (rows - 1)


def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


# The best configuration from find_best_pupil_config, using Chessboard 4 and a full grid homography.
best = calib.Config(
    distorted_camera_matrix=np.array(
        [
            [4681.74594217, 0.0, 980.69946863],
            [0.0, 5357.78011109, 604.19718367],
            [0.0, 0.0, 1.0],
        ]
    ),
    distortion_coefficients=np.array(
        [[-1.85381058, -43.40650075, 0.00164297, -0.05988727, 1313.59256105]]
    ),
    undistorted_camera_matrix=np.array(
        [
            [4452.88330078, 0.0, 945.37500454],
            [0.0, 4944.70849609, 604.61159019],
            [0.0, 0.0, 1.0],
        ]
    ),
    homography_matrix=np.array(
        [
            [1.09721761, 0.0749314, -458.46636981],
            [-0.01266892, 1.22193234, -37.49456483],
            [-0.00000743, 0.00012395, 1.0],
        ]
    ),
    grid_image_corners=calib.Corners(
        top_left=np.array([200.0, 200.0]),
        top_right=np.array([676.0, 200.0]),
        bottom_left=np.array([200.0, 880.0]),
        bottom_right=np.array([676.0, 880.0]),
    ),
    grid_space_corners=calib.Corners(
        top_left=np.array([0.0, 0.0], dtype=np.float32),
        top_right=np.array([318.83493, 0.0], dtype=np.float32),
        bottom_left=np.array([0.0, 455.4785], dtype=np.float32),
        bottom_right=np.array([318.83493, 455.4785], dtype=np.float32),
    ),
)


distances = []
distance_hist = {}
deviation_map = np.zeros((cols, rows), np.float32)

chess_image = cv.imread(
    "../find_best_pupil_config/sample_images/pupil_allignment_chessboard/distcor_004.bmp"
)
grey = cv.cvtColor(chess_image, cv.COLOR_BGR2GRAY)

_, distorted_grid = cv.findChessboardCorners(grey, (rows, cols))
corrected_points = calib.correct_points(
    distorted_grid, best, calib.Correction.lens_keystone_and_real_coordinates
)

# Determine each corrected points distance from the nearest point in a 0.5mm square grid
for lone_point in corrected_points:
    point = lone_point[0]
    x = int(round(point[0] / square_edge_mm))
    y = int(round(point[1] / square_edge_mm))
    expectation = (x * square_edge_mm, y * square_edge_mm)
    deviation = distance(point, expectation)
    distances.append(deviation)
    deviation_map[x, y] = deviation
    mm = math.ceil(deviation)
    if mm in distance_hist:
        distance_hist[mm] += 1
    else:
        distance_hist[mm] = 1

max_deviation = max(distances)

print("average deviation:\n{}mm".format(sum(distances) / len(distances)))
print("max deviation:\n{}mm".format(max_deviation))
print("deviation spread:")
pprint(distance_hist)

heatmap = np.zeros((rows, cols, 3), np.uint8)
for i in range(cols):
    for j in range(rows):
        heatmap[j, i, 2] = 255 * (deviation_map[i, j] / max_deviation)
cv.imshow(
    "corrected grid",
    calib.correct_image(chess_image, best, calib.Correction.lens_and_keystone),
)
cv.imwrite("pupil_heatmap.bmp", heatmap)
cv.imshow("pupil heatmap", heatmap)
cv.waitKey()
