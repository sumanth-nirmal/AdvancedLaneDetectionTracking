import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.signal import find_peaks_cwt
import pickle

images = glob.glob('camera_cal/calibration*.jpg')
for fname in images:
    img = cv2.imread(fname)

        # Undistort example calibration image
        img = mpimg.imread('camera_cal/calibration5.jpg')
        dst = cv2.undistort(img, camera_matrix, distortion_coefficients, None, camera_matrix)
        plt.imshow(dst)
        plt.savefig('example_images/undistort_calibration.png')
