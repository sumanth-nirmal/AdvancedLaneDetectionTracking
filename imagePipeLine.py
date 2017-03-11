from scipy import signal

import numpy as np
import cv2
import threshold
from line import Line
from moviepy.video.io.VideoFileClip import VideoFileClip
from threshold import threshold
from perspectiveTransform import warp, unwarp, warp_and_unwarp_params
from laneFinder import find_lines, paint_lane, composite_image, annotate
import time
import pickle


def find_lanes(image, LaneLines):
    # Main function
    t0 = time.time()

    # Undistort the image
    undist = cv2.undistort(image, find_lanes.mtx, find_lanes.dist)

    # Apply Sobel gradient threshold and HSL S-threshold
    stack = threshold(undist)

    # Warp the perspective to bird's eye
    image = warp(stack)

    # Find the lane lines (Note: need to implement previous result improvement and filtering!)
    lanes = find_lines(image, LaneLines)

    # Paint the lane lines onto the blank bird's eye image
    shadow = paint_lane(lanes, LaneLines)

    # Warp the bird's eye lane to original image perspective using inverse perspective matrix (Minv)
    unwarped = unwarp(shadow)

    # Overlay the lane estimate onto the original image
    dst = cv2.addWeighted(undist, 1, unwarped, 0.3, 0)
    annotated = annotate(dst, LaneLines)

    if find_lanes.composite is True:
        # combine images
        dst = composite_image(undist, stack, lanes, dst)

    return dst
