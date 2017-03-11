
#!/usr/bin/python
## Author: sumanth
## Date: March 08,2017
# test script for perspective transformation

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import cv2
import numpy as np
import threshold
from perspectiveTransform import warp, warp_and_unwarp_params
import glob

## testing on test images
images = glob.glob('test_images/straight_lines*.jpg')

## Load parameters ##
polygon1, polygon2 = warp_and_unwarp_params()
# perspective transform matrices
warp.M = cv2.getPerspectiveTransform(polygon1, polygon2)

count = 0;
for fname in images:
    img = cv2.imread(fname)
    count +=1

    # apply perspective correction
    warped = warp(img)
    plt.imshow(warped)
    plt.savefig('output_images/perspective/output' + str(count) + '.png')
    plt.gcf().clear()
