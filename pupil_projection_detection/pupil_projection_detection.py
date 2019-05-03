"""
# Purpose and background of this script:

Each FPU in the MOONS instrument is placed on a curved plate so that as the fibers move they will
always point at the same location, the pupil which the telescopes light passes through. This curve
has a radius of 4 meters. To test this, we project light through the fibers to a board four meters
away. As the FPU moves, we should expect the projected circle of light not to move, within a given
tolerance. This script aims to locate the center and size of those projected light circles in images
taken by the pupil alignment camera.

# Data source:

The pupil alignment camera images can be split into two batches, as between these the camera was
moved within the setup rig to have a larger section of its field of view occupied by the projection
board. The inital images are from two runs of the verification rig's program, and are located in the
image_dump and image_dump2 subfolders. Subsequent images are in the sample_images subfolder.

# Script development history:

Due to the uneven lighting and bleeding of light around the edges of the projected circle,
blobs found after otsu thresholding the images were visibly offset from the original circle.
This meant that the approach used in metrology_dot_detection wouldn't be applicable here.

Using a Hough Circle detector also seems unreliable, as the thresholds to identify the one true
circle vary across images.

Fortunately, using an adaptive local threshold technique does work to enhance the projected circles
edge, allowing it to be accurately located by a blob detector in images.

This was no longer the case after we repositioned the pupil alignment camera, as the new images
contained larger amounts of noise in the projection, which made the blob detector fail.
However, by introducing filtering steps to remove small black patches and black out the background
the white blob detector works across all pupil alignment images again.
"""
from __future__ import print_function
from itertools import chain
import math
import os
import cv2
import numpy as np
import sys


def show_image(image, title):
    cv2.imshow(title, image)
    cv2.moveWindow(title, 0, 0)


def clean_thresholded_image(image, noise_threshold, background_threshold, show=False):
    """
    Return a copy of the image with the background set to black and small black noise objects set to white
    :param image: The thresholded image to be cleaned
    :param noise_threshold: The maximum size a connected region can be to be counted as noise
    :param background_threshold: The smallest size a connected region can be to be counted as background
    :param show: Whether to display a graphic highlighting the alterations to be made
    :return: A cleaned copy of the thresholded image
    """

    clean_mean = image.copy()

    # connectedComponents() locates white regions on a black background.
    # Inverting allows us to locate black regions as distinct from eachother
    inverse_mean = cv2.bitwise_not(image)

    # Set any small black regions of noise to white
    # Noise is targeted as any small connected region of black in the thresholded image
    # Ensures the target blob doesn't contain smaller blobs inside it
    _, labels = cv2.connectedComponents(inverse_mean)
    unique_labels, counts = np.unique(labels, return_counts=True)
    # index zero is the white of the original image, so won't need its colour changing
    noise_labels = np.extract(counts[1:] < noise_threshold, unique_labels[1:])
    noise_condition = np.isin(labels, noise_labels)
    # set any pixels in those small black regions to white
    np.putmask(clean_mean, noise_condition, 255)

    # Set the background to black
    # The background is targeted as any large connected region of white in the thresholded image
    # Prevents the target circle being a concentric blob inside the background
    _, labels = cv2.connectedComponents(image)
    unique_labels, counts = np.unique(labels, return_counts=True)
    # index zero is the black of the image, so won't need its colour changing
    bg_labels = np.extract(counts[1:] > background_threshold, unique_labels[1:])
    bg_condition = np.isin(labels, bg_labels)
    # set any pixels in those large white regions to black
    np.putmask(clean_mean, bg_condition, 0)

    if show:
        cleaning_steps = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        # Highlight noise pixels in red
        np.putmask(cleaning_steps[:, :, 0], noise_condition, 0)
        np.putmask(cleaning_steps[:, :, 1], noise_condition, 0)
        np.putmask(cleaning_steps[:, :, 2], noise_condition, 255)
        # Hightlight background pixels in blue
        np.putmask(cleaning_steps[:, :, 0], bg_condition, 255)
        np.putmask(cleaning_steps[:, :, 1], bg_condition, 0)
        np.putmask(cleaning_steps[:, :, 2], bg_condition, 0)
        show_image(cleaning_steps, "mean cleaning steps (blue -> black, red -> white)")

    return clean_mean


def find_white_circles(image, min_area, max_area):
    """
    Return a list of white circles found in the image
    :param image: The image to search for circles in
    :param min_area: The minimum area a circle should have to be accepted
    :param max_area: The maximum area a circle should have to be accepted
    :return: A list of (center x, center y, radius) tuples for each circle in the image
    """
    params = cv2.SimpleBlobDetector_Params()
    params.minArea = min_area
    params.maxArea = max_area
    params.blobColor = 255  # white
    params.filterByColor = True
    params.minInertiaRatio = 0.9  # non stretched
    params.filterByInertia = True
    params.minConvexity = 0.9  # convex
    params.filterByConvexity = True
    params.filterByCircularity = False
    # Due to the noisy edge, circularity filtering gives false negatives, so isn't done here

    detector = cv2.SimpleBlobDetector_create(params)

    return [
        (blob.pt[0], blob.pt[1], blob.size / 2.0) for blob in detector.detect(image)
    ]


def detect_pupil_projection_circles(image_path, show=False):
    minradius = 150
    maxradius = 250
    min_area = math.pi * minradius ** 2
    max_area = math.pi * maxradius ** 2

    image = cv2.imread(image_path)
    greyscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Enhance the edges with a locally adaptive threshold
    mean = cv2.adaptiveThreshold(
        greyscale, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 81, 2
    )
    if show:
        show_image(image, "original image")
        show_image(mean, "mean")

    # Remove the bright background and noise in the projected light circle
    clean_mean = clean_thresholded_image(mean, 5000, max_area, show)

    circles = find_white_circles(clean_mean, min_area, max_area)

    if show:
        int_circles = np.round(np.array(circles)).astype("int")
        results = image.copy()
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in int_circles:
            # draw concentric circles around the circles edge
            cv2.circle(results, (x, y), r + 15, (0, 255, 0), 1)
            cv2.circle(results, (x, y), r - 15, (0, 255, 0), 1)
            cv2.circle(results, (x, y), r + 5, (0, 64, 128), 1)
            cv2.circle(results, (x, y), r - 5, (0, 64, 128), 1)
            # draw a square around the center of the circle
            cv2.rectangle(results, (x + 1, y + 1), (x - 1, y - 1), (0, 0, 255), 1)
            print(x, y, r)

        show_image(clean_mean, "clean mean")
        show_image(results, "results")

    return circles


if __name__ == "__main__":
    checked = 0

    locations = ("sample_images/pupil_projections", "image_dump", "image_dump2")

    for root, directories, files in chain.from_iterable(
        os.walk(location) for location in locations
    ):
        for filename in files:
            path = os.path.join(root, filename)
            if ("pupil" in path) and os.stat(path).st_size > 0:
                print(path)

                projected_circles = detect_pupil_projection_circles(path, show=True)

                if len(projected_circles) == 0:
                    print(
                        "No fiber projections detected in image {}".format(path),
                        file=sys.stderr,
                    )
                elif len(projected_circles) > 1:
                    print(
                        "Multiple fiber projections detected in image {}".format(path),
                        file=sys.stderr,
                    )
                cv2.waitKey()
                checked += 1

    print("{} images checked".format(checked))
    cv2.waitKey()
