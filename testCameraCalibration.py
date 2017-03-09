#!/usr/bin/python
## Author: sumanth
## Date: March 08,2017
# test script for camera calibration

import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.signal import find_peaks_cwt
import pickle

images = glob.glob('camera_cal/calibration*.jpg')

file = open("cameraCalibrationParams.pickle",'rb')
object_file = pickle.load(file)
file.close()

count = 0;
for fname in images:
    img = cv2.imread(fname)
    count +=1
    dst = cv2.undistort(img, object_file['mtx'], object_file['dist'], None, object_file['mtx'])
    plt.imshow(dst)
    cv2.imshow('img',dst)
    #cv2.waitKey(1200)
    # save the corrected images
    plt.savefig('corrected_images/corrected_calibration' + str(count) + '.png')
