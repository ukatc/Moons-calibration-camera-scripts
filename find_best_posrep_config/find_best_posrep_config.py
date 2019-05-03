"""
For the moons FPU verification we needed to make highly accurate measurements of FPU metrology dots
in real world units from images taken by the positional repeatability camera. This need drove the
development of the camera_calibration package.

This script iterates through all available combinations of dot grid images and homography settings.
For each one it builds a calibration configuration and determines how accurate its pixel space to
calibration plane real space transform is, printing the results to console. This script was used to
determine that the best accuracy calibration config would be built using distcor04 and full grid
homography. Results of its output can be found in the Calibration Configuration Accuracies appendix
of the "Developing the calibration package and configuration" report in the camera_calibration repo.

It should be noted that this script takes a very long time to run (~8 minutes for each of the 11
grid images on an i7-6700HQ @ 3.1GHz).
"""
from __future__ import print_function
import camera_calibration as calib
import cv2 as cv
import math
import numpy as np
from pprint import pprint
import sys

rows = 116
cols = 170

np.set_printoptions(suppress=True)


def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def assess_config(image_path, dot_image, distorted_grid, corner_only_homography):
    sparse_rows = rows // 2
    sparse_cols = cols // 2
    sparse_grid = np.zeros((sparse_rows * sparse_cols, 1, 2), np.float32)
    for i in range(sparse_rows):
        for j in range(sparse_cols):
            sparse_grid[i * sparse_cols + j, 0, 0] = distorted_grid[
                (i * 2) * cols + (j * 2), 0, 0
            ]
            sparse_grid[i * sparse_cols + j, 0, 1] = distorted_grid[
                (i * 2) * cols + (j * 2), 0, 1
            ]

    h, w = dot_image.shape[:2]
    dot_config = calib.Config()
    dot_config.populate_lens_parameters_from_grid(
        sparse_grid, sparse_cols, sparse_rows, w, h
    )

    undistorted_distorted_grid = cv.undistortPoints(
        distorted_grid,
        dot_config.distorted_camera_matrix,
        dot_config.distortion_coefficients,
        P=dot_config.undistorted_camera_matrix,
    )

    dot_config.populate_keystone_and_real_parameters_from_grid(
        undistorted_distorted_grid,
        cols,
        rows,
        84.5,
        57.5,
        corners_only=corner_only_homography,
    )

    expectations = np.zeros((len(distorted_grid), 2), np.float32)
    for i in range(rows):
        for j in range(cols):
            expectations[i * cols + j, 0] = 84.5 - (0.5 * j)
            expectations[i * cols + j, 1] = 57.5 - (0.5 * i)

    corrected_points = calib.correct_points(
        distorted_grid, dot_config, calib.Correction.lens_keystone_and_real_coordinates
    )

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
        microns = math.ceil(dist * 1000)
        if microns in distance_hist:
            distance_hist[microns] += 1
        else:
            distance_hist[microns] = 1

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

    chess_config = calib.Config()
    chess_config.populate_lens_parameters_from_chessboard(
        "sample_images/002h.bmp", 6, 8
    )

    params = cv.SimpleBlobDetector_Params()
    params.minArea = 50
    params.maxArea = 1000
    params.filterByArea = True
    params.minCircularity = 0.2
    params.filterByCircularity = True
    params.blobColor = 0
    params.filterByColor = True
    dot_detector = cv.SimpleBlobDetector_create(params)

    for image_path in (
        "sample_images/distcor_01_cleaned.bmp",
        "sample_images/distcor_02_cleaned.bmp",
        "sample_images/distcor_03_cleaned.bmp",
        "sample_images/distcor_04_cleaned.bmp",
        "sample_images/distcor_05_cleaned.bmp",
        "sample_images/distcor_06_cleaned.bmp",
        "sample_images/distcor_07_cleaned.bmp",
        "sample_images/distcor_08_cleaned.bmp",
        "sample_images/distcor_09_cleaned.bmp",
        "sample_images/distcor_10_cleaned.bmp",
        "sample_images/distcor_11_cleaned.bmp",
    ):
        dot_image = cv.imread(image_path)
        undistorted_dot_image = calib.correct_image(
            dot_image, chess_config, calib.Correction.lens_distortion
        )

        found, undistorted_grid = cv.findCirclesGrid(
            undistorted_dot_image,
            (cols, rows),
            cv.CALIB_CB_SYMMETRIC_GRID + cv.CALIB_CB_CLUSTERING,
            dot_detector,
            cv.CirclesGridFinderParameters(),
        )

        if not found:
            print("Could not find dot grid in {}".format(image_path))
            continue

        distorted_points = np.array(
            [[point] for point in cv.KeyPoint_convert(dot_detector.detect(dot_image))],
            np.float32,
        )
        transformed_points = cv.undistortPoints(
            distorted_points,
            chess_config.distorted_camera_matrix,
            chess_config.distortion_coefficients,
            P=chess_config.undistorted_camera_matrix,
        )
        distorted_grid = np.zeros(undistorted_grid.shape, undistorted_grid.dtype)
        for i in range(len(undistorted_grid)):
            if i % 100 == 0:
                print("progress: {}/{}".format(i, cols * rows), end="\r")
            # get the point at i in the grid
            grid_member = undistorted_grid[i]
            # find the nearest member of transformed_points
            nearest_distance = sys.float_info.max
            original_point = None
            for j in range(len(transformed_points)):
                transformed_point = transformed_points[j]
                separation = distance(grid_member[0], transformed_point[0])
                if separation < nearest_distance:
                    nearest_distance = separation
                    original_point = distorted_points[j]
                if separation < 1:
                    break
            # get the untransformed point that matches the transformed_point
            assert original_point is not None
            # set it to position i in the new grid
            distorted_grid[i, 0, 0] = original_point[0, 0]
            distorted_grid[i, 0, 1] = original_point[0, 1]

        for corner_only_homography in (True, False):
            assess_config(image_path, dot_image, distorted_grid, corner_only_homography)


if __name__ == "__main__":
    search_for_best_config()
