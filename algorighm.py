# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 20:11:53 2020

@author: nlutz
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


def minimum_bounding_rectangle(points):
    """
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.

    :param points: an nx2 matrix of coordinates
    :rval: an nx2 matrix of coordinates
    """
    from scipy.ndimage.interpolation import rotate
    pi2 = np.pi/2.

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = np.zeros((len(hull_points)-1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    # XXX both work
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles-pi2),
        np.cos(angles+pi2),
        np.cos(angles)]).T
#     rotations = np.vstack([
#         np.cos(angles),
#         -np.sin(angles),
#         np.sin(angles),
#         np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    return rval

img_path = "images/1.jpg"

im_orig = cv2.imread(img_path)
im = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)


# Blur image
blurred = cv2.GaussianBlur(im, (9,9), 10)

unsharp_image = cv2.addWeighted(im, 1.5, blurred, -0.5, 0, im)



blurred = cv2.medianBlur(unsharp_image, 9)

# Convert image to binary
ret,thresh1 = cv2.threshold(blurred,70,255,cv2.THRESH_BINARY)
#plt.imshow(thresh1)
_, contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

cv2.drawContours(im_orig, contours, -1, (0,255,0), 1)


for c in contours:
    
    c = c.reshape((c.shape[0], c.shape[2]))
    rval = minimum_bounding_rectangle(c)
    
    for coordinate in rval:
        a = im_orig[int(coordinate[1])][int(coordinate[0])]
        im_orig[int(coordinate[1])][int(coordinate[0])]=[255,255,255]




#edges = cv2.Canny(thresh1,0,1)

plt.imshow(im_orig)

