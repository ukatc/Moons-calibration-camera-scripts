"""
An adapted copy of find_best_posrep_config, this script produces correction configurations for the
positional repeatability camera, testing their accuracy and printing the results.

However, as OpenCV can detect the grid directly from the image, there is no reverse grid mapping
step, so unlike the posrep script runtime is not a concern for this one.

The best calibration found uses grid image discor_004.bmp with a full grid homography, giving an
average deviation of 0.36mm with max deviation of 0.83mm
"""
from __future__ import division, print_function
import camera_calibration as calib
import cv2 as cv
import math
import numpy as np
from pprint import pprint

rows = 11
cols = 8

# measured distances for various numbers of square edges on the printed calibration image
lengths_mm = [45.65, 91.13, 136.58, 64.30, 128.78]
lengths_edges = [1, 2, 3, math.sqrt(2), math.sqrt(2) * 2]

square_edge_mm = sum(
    lengths_mm[i] / lengths_edges[i] for i in range(len(lengths_mm))
) / len(lengths_mm)

grid_width = square_edge_mm * (cols - 1)
grid_height = square_edge_mm * (rows - 1)

print("edge:{}\nwidth:{}\nheight:{}".format(square_edge_mm, grid_width, grid_height))
print(
    "deviations:{}".format(
        [
            (square_edge_mm - (lengths_mm[i] / lengths_edges[i])) / square_edge_mm
            for i in range(len(lengths_mm))
        ]
    )
)


np.set_printoptions(suppress=True)


def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def assess_config(image_path, chess_image, distorted_grid, corner_only_homography):

    config = calib.Config()
    config.populate_lens_parameters_from_chessboard(chess_image, rows, cols)
    config.populate_keystone_and_real_parameters_from_chessboard(
        chess_image,
        cols,
        rows,
        grid_width,
        grid_height,
        corners_only=corner_only_homography,
        border=200,
    )

    cv.imshow(
        "{} {} lens".format(image_path, corner_only_homography),
        calib.correct_image(chess_image, config, calib.Correction.lens_distortion),
    )
    cv.imshow(
        "{} {} lens and keystone".format(image_path, corner_only_homography),
        calib.correct_image(chess_image, config, calib.Correction.lens_and_keystone),
    )
    # grid[0] -> bottom left
    # grid[10] -> top left
    # grid[-11] -> bottom right
    # grid[-1] -> top right
    expectations = np.zeros((len(distorted_grid), 2), np.float32)
    for i in range(rows):
        for j in range(cols):
            expectations[i + j * rows, 0] = square_edge_mm * j
            expectations[i + j * rows, 1] = grid_height - (square_edge_mm * i)

    corrected_points = calib.correct_points(
        distorted_grid, config, calib.Correction.lens_keystone_and_real_coordinates
    )

    print(config)
    print(config.to_dict())

    distances = []
    distance_hist = {}
    highest = 0
    for i in range(len(corrected_points)):
        original = distorted_grid[i, 0]
        point = corrected_points[i, 0]
        expectation = expectations[i]
        assert len(point) == 2
        dist = distance(point, expectation)
        if dist > highest:
            highest = dist
            print(
                "new highest: px{} res{} exp{} dist{}".format(
                    original, point, expectation, dist
                )
            )
        distances.append(dist)
        mm = math.ceil(dist)
        if mm in distance_hist:
            distance_hist[mm] += 1
        else:
            distance_hist[mm] = 1

    print(
        "image: {} corners_only_homography: {}".format(
            image_path, corner_only_homography
        )
    )
    print("average deviation:\n{}mm".format(sum(distances) / len(distances)))
    print("max deviation:\n{}mm".format(max(distances)))
    print("deviation spread:")
    pprint(distance_hist)


def search_for_best_config():

    for image_path in (
        "sample_images/pupil_allignment_chessboard/distcor_001.bmp",
        "sample_images/pupil_allignment_chessboard/distcor_002.bmp",
        "sample_images/pupil_allignment_chessboard/distcor_003.bmp",
        "sample_images/pupil_allignment_chessboard/distcor_004.bmp",
    ):
        chess_image = cv.imread(image_path)
        grey = cv.cvtColor(chess_image, cv.COLOR_BGR2GRAY)

        _, distorted_grid = cv.findChessboardCorners(grey, (rows, cols))

        for corner_only_homography in (True, False):
            assess_config(
                image_path, chess_image, distorted_grid, corner_only_homography
            )

        cv.drawChessboardCorners(chess_image, (rows, cols), distorted_grid, True)
        cv.imshow("{} grid".format(image_path), chess_image)

    cv.waitKey()


if __name__ == "__main__":
    search_for_best_config()
