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
import glob

img_file = 'test_images/test6.jpg'

# fetch the camera p[arameters
with open('cameraCalibrationParams.pickle', 'rb') as f:
    save_dict = pickle.load(f)
mtx = save_dict['mtx']
dist = save_dict['dist']

##### Option -1 thresholding
# load a test image
img = mpimg.imread(img_file)
img = cv2.undistort(img, mtx, dist, None, mtx)

combined, abs_bin, mag_bin, dir_bin, hls_bin = threshold.imageThreshold(img)

s=plt.subplot(2, 3, 1)
plt.imshow(img)
s.set_title("inputImage")

x = plt.subplot(2, 3, 2)
plt.imshow(abs_bin, cmap='gray', vmin=0, vmax=1)
x.set_title("absSobelThrImage")

y=plt.subplot(2, 3, 3)
plt.imshow(mag_bin, cmap='gray', vmin=0, vmax=1)
y.set_title("magThrImage")

z=plt.subplot(2, 3, 4)
plt.imshow(dir_bin, cmap='gray', vmin=0, vmax=1)
z.set_title("DirThrImage")

r=plt.subplot(2, 3, 5)
plt.imshow(hls_bin, cmap='gray', vmin=0, vmax=1)
r.set_title("HLSThImage")

e=plt.subplot(2, 3, 6)
plt.imshow(combined, cmap='gray', vmin=0, vmax=1)
e.set_title("finalImage")

plt.savefig('corrected_images/thresholded_subplot.png')

plt.tight_layout()
plt.show()
plt.gcf().clear()


#### option - 2
# generate for all the test images
xgrad_thresh_temp = (40,100)
s_thresh_temp=(150,255)

images = glob.glob('test_images/*.jpg')

count = 0;
for fname in images:
    img = cv2.imread(fname)
    count +=1
    combined_binary = threshold.applyThresholdColorHLS(img, xgrad_thresh=xgrad_thresh_temp, s_thresh=s_thresh_temp)
    plt.imshow(combined_binary)
    plt.savefig('corrected_images/threshold/image' + str(count) + '.png')
    plt.gcf().clear()
