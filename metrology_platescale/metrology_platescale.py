"""
The metrology camera used in the MOONS FPU verification rig has a lens setup with no distortion that
captures an area of constant width regardless of the distance from the camera to the target. This
means that when aimed directly along the normal of a target plane, the relative location of points
in the imaged plane can be calculated by knowing the platescale of the camera in px/mm.

This script analyses three images of a precisely manufactured glass etched dot grids to determine
the camera's platescale.
"""
import cv2 as cv
import math


def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


minradius = 10
maxradius = 20

# Create a blob detector that can identify the dots in the glass etched grid
params = cv.SimpleBlobDetector_Params()
params.minArea = math.pi * minradius ** 2
params.maxArea = math.pi * maxradius ** 2
params.blobColor = 0  # black
params.filterByColor = True

detector = cv.SimpleBlobDetector_create(params)

# The centers of dots in the grid are separated by 0.5mm, with 26 rows and columns of dots
edge_mm = 12.5
diagonal_mm = distance((0, 0), (edge_mm, edge_mm))

averages = []

base = "metcal_platescale_00{}.bmp"
for path in [base.format(i) for i in range(1, 4)]:
    print(path)
    img = cv.imread(path)
    # Locate the grid in the image
    found, grid = cv.findCirclesGrid(
        img,
        (26, 26),
        cv.CALIB_CB_SYMMETRIC_GRID + cv.CALIB_CB_CLUSTERING,
        detector,
        cv.CirclesGridFinderParameters(),
    )

    if not found:
        print("grid not found")
        blobs = detector.detect(img)
        print("blobs detected: {}".format(len(blobs)))
        break

    bottomright = grid[0, 0]
    bottomleft = grid[25, 0]
    topleft = grid[-1, 0]
    topright = grid[-26, 0]

    # Calculate the platescale using known distances in the grid
    distances = [
        distance(topleft, topright) / edge_mm,
        distance(topleft, bottomleft) / edge_mm,
        distance(topright, bottomright) / edge_mm,
        distance(bottomleft, bottomright) / edge_mm,
        distance(topleft, bottomright) / diagonal_mm,
        distance(topright, bottomleft) / diagonal_mm,
    ]
    average = sum(distances) / len(distances)
    distances.append(average)
    averages.append(average)

    output = """Top edge: {}
Left edge: {}
Right edge: {}
Bottom edge: {}
TL-BR diagonal: {}
TR-BL diagonal: {}
average: {}
""".format(
        *distances
    )

    print(output)

print(
    "total average: {}\nAll measurements in px/mm".format(sum(averages) / len(averages))
)
