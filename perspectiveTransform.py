#!/usr/bin/python
## Author: sumanth
## Date: March 08,2017
# methods for perspective transformation

import cv2
import numpy as np
def warp(src):
    # given an undistorted image, create a bird's eye transformation
    # of the area in front of the car

    rows, cols = src.shape[0], src.shape[1]
    dst = cv2.warpPerspective(src, warp.M, (cols, rows))

    return dst

def unwarp(src):
    # given a bird's eye view of the area in front of the car,
    # create a perspective transformation (natural view)

    rows, cols = src.shape[0], src.shape[1]
    dst = cv2.warpPerspective(src, unwarp.Minv, (cols, rows))

    return dst

def warp_and_unwarp_params():
    # Constant parameters for warp and unwarp

    cols = np.float32(1280)
    rows = np.float32(720)
    horz = np.float32(450) # horizon y-coordinate
    center = np.float32(cols/2) # horizontal center x-coordinate
    tr_width = np.float32(200) # width of the trapezoid upper leg
    s = np.float32(0.3) # slope of the trapezoid right leg (dy/dx)

    p1 = [center-tr_width/2, horz] # upper left vertex
    p4 = [center+tr_width/2, horz] # upper right vertex
    p2 = [p1[0]-(rows-horz)/s, rows] # lower left vertex
    p3 = [p4[0]+(rows-horz)/s, rows] # lower right vertex

    # warp polygon
    poly1 = np.float32([p1,p2,p3,p4])

    # result polygon (image border)
    poly2 = np.float32([[0,0],[0,rows],[cols,rows],[cols,0]])

    return poly1, poly2
