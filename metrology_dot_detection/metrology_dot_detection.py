"""
To determine the position of an FPU in images taken by the Positional repeatability and metrology
calibration cameras, each FPU is fitted with a pair of differently sized metrology targets. These
round white circles are of known sizes, are a known distance apart, and have high contrast against
the black material of the FPU's and most of the verification rig. However, the light that
illuminates these targets can also reflect off other components of the FPU's and the rig, so a
detection method is needed that reliably locates these targets, while ignoring false positives.

This detection is achieved using an openCV blob detector that filters its results on shape, colour
and size. This detector is then run again on an OTSU thresholded copy of the image, and blobs that
were found with greatly varying locations and sizes are discarded, effectively filtering on the
sharpness of the blob's edge. Finally, blobs are ignored if they're located more than a certain
distance from other blobs.

This method has been tested on 601 images from the positional repeatability camera, and 383 images
from the metrology calibration camera, correctly identifying only the metrology dots (and occasional
lit fibers) in all of them.

As the different cameras have different zoom levels, each has different parameters that are passed
to the dot detection method.

This script was initially developed against a small number of images, located in the metrology_dots
subfolder. The majority of the images (the 984 afformentioned ones) were produced by full runs of
the verification rig test program. Samples of these are included in the image_dump and image_dump2
subfolders.
"""
from __future__ import print_function
import numpy as np
import cv2
import math
import os
from itertools import chain


def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def find_bright_sharp_circles(path, minradius, maxradius, grouprange=None, show=False):
    """
    Finds circular dots in the given image within the radius range, displaying them on console and graphically if show is set to True

    Works by detecting white circular blobs in the raw image, and in an otsu thresholded copy.
    Circles that have similar center locations and radii in both images are kept.

    Setting grouprange to a number will mean only circles within that distance of other circles are returned,
    however if only 1 is found, it will be returned and grouprange will have no effect.
    :return: a list of (center_x, center_y, radius) tuples for each detected dot.
    """
    image = cv2.imread(path)
    greyscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(greyscale, (5, 5), 0)
    retval, thresholded = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    output = image.copy()

    params = cv2.SimpleBlobDetector_Params()
    params.minArea = math.pi * minradius ** 2
    params.maxArea = math.pi * maxradius ** 2
    params.blobColor = 255  # white
    params.filterByColor = True
    params.minCircularity = 0.4  # smooth sided (0 is very pointy)
    params.filterByCircularity = True
    params.minInertiaRatio = 0.9  # non stretched
    params.filterByInertia = True
    params.minConvexity = 0.9  # convex
    params.filterByConvexity = True

    detector = cv2.SimpleBlobDetector_create(params)

    # blob detect on the original
    blobs = detector.detect(image)

    # blob detect on the thresholded copy
    binary_blobs = detector.detect(thresholded)

    # keep blobs found in both with similar sizes
    if show:
        print(path)
        print("round blobs in original:")
        print([(blob.pt[0], blob.pt[1], blob.size / 2.0) for blob in blobs])
        print("round blobs after otsu thresholding:")
        print([(blob.pt[0], blob.pt[1], blob.size / 2.0) for blob in binary_blobs])

    circles = []
    for blob in blobs:
        for binary_blob in binary_blobs:
            # They match if center's and radii are similar
            if (
                distance(blob.pt, binary_blob.pt)
                + abs(blob.size / 2.0 - binary_blob.size / 2.0)
                < 6.0
            ):
                circles.append((blob.pt[0], blob.pt[1], blob.size / 2.0))
                print(
                    distance(blob.pt, binary_blob.pt)
                    + abs(blob.size / 2.0 - binary_blob.size / 2.0)
                )
                break

    if not (grouprange is None or len(circles) == 1):
        accepted = []
        for i in range(len(circles)):
            for j in range(len(circles)):
                if i != j:
                    print(distance(circles[i], circles[j]))
                    if distance(circles[i], circles[j]) < grouprange:
                        accepted.append(circles[i])
                        break
        circles = accepted

    if show:
        width, height = image.shape[1] // 4, image.shape[0] // 4
        shrunk_original = cv2.resize(image, (width, height))
        # ensure at least some circles were found
        if circles is not None:
            # convert the (x, y) coordinates and radius of the circles to integers
            print("matching points:")
            print(circles)
            int_circles = np.round(np.array(circles)).astype("int")
            # loop over the (x, y) coordinates and radius of the circles
            for (x, y, r) in int_circles:
                # draw the circle in the output image, then draw a rectangle
                # corresponding to the center of the circle
                cv2.circle(output, (x, y), r, (0, 255, 0), 4)
                cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

        # show the output image
        shrunk_output = cv2.resize(output, (width, height))
        shrunk_thresh = cv2.resize(
            cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR), (width, height)
        )
        cv2.imshow(path, np.hstack([shrunk_original, shrunk_output, shrunk_thresh]))
        cv2.moveWindow(path, 0, 0)
        print()

    return circles


if __name__ == "__main__":
    path, maxr, minr = "metrology_dots/PT25_metcal_1_00{}.bmp", 200, 45
    paths = [path.format(i) for i in range(1, 6)]
    for path in paths:
        if os.path.isfile(path):
            find_bright_sharp_circles(path, minr, maxr, 525, show=True)
            cv2.waitKey()
            cv2.destroyWindow(path)

    path, maxr, minr = "metrology_dots/PT25_posrep_1_00{}.bmp", 55, 15
    paths = [path.format(i) for i in range(1, 6)]
    for path in paths:
        if os.path.isfile(path):
            find_bright_sharp_circles(path, minr, maxr, 200, show=True)
            cv2.waitKey()
            cv2.destroyWindow(path)

    path, maxr, minr = "metrology_dots/PT24_posrep_selftest.bmp", 55, 15
    if os.path.isfile(path):
        find_bright_sharp_circles(path, minr, maxr, 200, show=True)
        cv2.waitKey()
        cv2.destroyWindow(path)

    locations = ("image_dump", "image_dump2")

    posreps = 0
    empty = 0
    metcal = 0
    height = 0
    pupil = 0
    other = []
    for root, directories, files in chain.from_iterable(
        os.walk(location) for location in locations
    ):
        for file in files:
            path = os.path.join(root, file)
            if os.stat(path).st_size == 0:
                empty += 1
            elif "posrep" in path or "positional" in path:
                posreps += 1
            elif "metcal" in path or "datumed" in path or "met-cal-target" in path:
                metcal += 1
            elif "metrology-height" in path:
                height += 1
            elif "pupil" in path:
                pupil += 1
            else:
                other.append(path)

    print(other, len(other), posreps, metcal, empty, height, pupil)

    opened = 0
    checked = 0
    for root, directories, files in chain.from_iterable(
        os.walk(location) for location in locations
    ):
        for file in files:
            path = os.path.join(root, file)
            if ("posrep" in path or "positional" in path) and os.stat(path).st_size > 0:
                circles = find_bright_sharp_circles(
                    path, 15, 55, grouprange=200, show=True
                )
                cv2.waitKey()
                cv2.destroyWindow(path)
                if len(circles) != 2:
                    print(path, circles)
                    opened += 1
                checked += 1
    print(checked, opened)

    opened = 0
    checked = 0
    for root, directories, files in chain.from_iterable(
        os.walk(location) for location in locations
    ):
        for file in files:
            path = os.path.join(root, file)
            if (
                "metcal" in path or "datumed" in path or "met-cal-target" in path
            ) and os.stat(path).st_size > 0:
                circles = find_bright_sharp_circles(
                    path, 45, 200, grouprange=525, show=True
                )
                cv2.waitKey()
                cv2.destroyWindow(path)
                if len(circles) != 2:
                    print(path, circles)
                    opened += 1
                checked += 1
    print(checked, opened)
