
#!/usr/bin/python
## Author: sumanth
## Date: March 08,2017
# file implementing the pipline for lane dettection and tracking

from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import cv2
import threshold
from line import Line
from moviepy.video.io.VideoFileClip import VideoFileClip
from threshold import threshold
from perspectiveTransform import warp, unwarp, warp_and_unwarp_params
from laneFinder import find_lines, paint_lane, composite_image, annotate
import pickle


def find_lanes(image, LaneLines, vis=False):
    # Undistort the image
    undist = cv2.undistort(image, find_lanes.mtx, find_lanes.dist)
    # visualisation
    if vis == True:
        plt.clf()
        plt.imshow(undist)
        plt.savefig('output_images/pipeline/undistort.jpg')
        plt.clf()
        plt.imshow(image)
        plt.savefig('output_images/pipeline/input.jpg')


    # Apply Sobel gradient threshold and HSL S-threshold
    stack = threshold(undist)
    # visualisation
    if vis == True:
        plt.clf()
        plt.imshow(stack)
        plt.savefig('output_images/pipeline/threshold.jpg')

    # Warp the perspective to bird's eye
    image = warp(stack)
    # visualisation
    if vis == True:
        plt.clf()
        plt.imshow(image)
        plt.savefig('output_images/pipeline/wrapped.jpg')

    # Find the lane lines (Note: need to implement previous result improvement and filtering!)
    lanes = find_lines(image, LaneLines)
    # visualisation
    if vis == True:
        plt.clf()
        plt.imshow(lanes)
        plt.savefig('output_images/pipeline/lanes.jpg')

    # Paint the lane lines onto the blank bird's eye image
    shadow = paint_lane(lanes, LaneLines)
    # visualisation
    if vis == True:
        plt.clf()
        plt.imshow(shadow)
        plt.savefig('output_images/pipeline/laneOnBirdEye.jpg')

    # Warp the bird's eye lane to original image perspective using inverse perspective matrix (Minv)
    unwarped = unwarp(shadow)
    # visualisation
    if vis == True:
        plt.clf()
        plt.imshow(unwarped)
        plt.savefig('output_images/pipeline/unwrapped.jpg')

    # Overlay the lane estimate onto the original image
    dst = cv2.addWeighted(undist, 1, unwarped, 0.3, 0)
    annotated = annotate(dst, LaneLines)
    # visualisation
    if vis == True:
        plt.clf()
        plt.imshow(annotated)
        plt.savefig('output_images/pipeline/annotated.jpg')

    if find_lanes.composite is True:
        # combine images
        dst = composite_image(undist, stack, lanes, dst)

    return dst
