#!/usr/bin/python
## Author: sumanth
## Date: March 08,2017
# test script for thresholding techniques

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import threshold

img_file = 'test_images/test6.jpg'

# fetch the camera p[arameters
with open('cameraCalibrationParams.pickle', 'rb') as f:
	save_dict = pickle.load(f)
mtx = save_dict['mtx']
dist = save_dict['dist']

# load a test image
img = mpimg.imread(img_file)
img = cv2.undistort(img, mtx, dist, None, mtx)

combined, abs_bin, mag_bin, dir_bin, hls_bin = threshold.imageThreshold(img)

plt.subplot(2, 3, 1)
plt.imshow(abs_bin, cmap='gray', vmin=0, vmax=1)
plt.subplot(2, 3, 2)
plt.imshow(mag_bin, cmap='gray', vmin=0, vmax=1)
plt.subplot(2, 3, 3)
plt.imshow(dir_bin, cmap='gray', vmin=0, vmax=1)
plt.subplot(2, 3, 4)
plt.imshow(hls_bin, cmap='gray', vmin=0, vmax=1)
plt.subplot(2, 3, 5)
plt.imshow(img)
plt.subplot(2, 3, 6)
plt.imshow(combined, cmap='gray', vmin=0, vmax=1)

plt.tight_layout()
plt.show()
