#!/usr/bin/python
## Author: sumanth
## Date: March 08,2017
# test script for thresholding techniques

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from threshold import threshold, SobelX, HlsGrad, ColorFilt
import glob

with open('cameraCalibrationParams.pickle', 'rb') as f:
	save_dict = pickle.load(f)
mtx = save_dict['mtx']
dist = save_dict['dist']

# find_lanes.mtx = mtx
# find_lanes.dist = dist

# Sobel and HLS thresholds
SobelX.thresh, HlsGrad.s_thresh = [20, 100], [120, 225]
# Yellow and white thresholds
ColorFilt.yellow, ColorFilt.white = [[215, 255], [140, 255], [0, 160]],[[225, 255], [225, 255], [225, 255]]

images = glob.glob('test_images/*.jpg')

count = 0;
for fname in images:
    img = cv2.imread(fname)
    count +=1
    combined_binary = threshold(img)
    plt.imshow(combined_binary, cmap="gray")
    plt.savefig('output_images/threshold/image' + str(count) + '.png')
    plt.gcf().clear()
