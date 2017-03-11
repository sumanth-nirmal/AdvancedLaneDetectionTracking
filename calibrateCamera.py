#!/usr/bin/python
## Author: sumanth
## Date: March 08,2017
# script to calibrate the camera based on the checker board

import argparse
import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.signal import find_peaks_cwt
import pickle
#%matplotlib inline

def calibrateCamera(path, tp, x, y):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    #fetch the images
    images = glob.glob(path+'*.jpg')
    # Step through the list and search for chessboard corners
    count=0
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # detect the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (x,y),None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            count += 1
            img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
            plt.clf()
            plt.imshow(img)
            plt.savefig('output_images/calibrated/output' + str(count) + '.jpg')

    cv2.destroyAllWindows()

    # claibrate the camera
    img = cv2.imread(tp)
    img_size = (img.shape[1], img.shape[0])
    ret, cam_mtx, dist_coff, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

    return cam_mtx, dist_coff


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Camera Calibration')
    parser.add_argument(
        'path',
        type=str,
        default='camera_cal/calibration',
        help='Path where the images for calibration are stored, for instance the checker board images'
    )
    parser.add_argument(
        'testimage',
        type=str,
        default='test_images/straight_lines1.jpg',
        help='give a test image for which the camera should be claibrated'
    )
    parser.add_argument(
        'x',
        type=int,
        default=9,
        help='The number of inside corners in x'
    )
    parser.add_argument(
        'y',
        type=int,
        default=6,
        help='The number of inside corners in y'
    )
    args = parser.parse_args()

    camera_matrix, distortion_coefficients = calibrateCamera(args.path, args.testimage, args.x, args.y)

    # save the calibration paramters
    save_dict = {'mtx': camera_matrix, 'dist': distortion_coefficients}
    with open('cameraCalibrationParams.pickle', 'wb') as f:
        pickle.dump(save_dict, f)
        print('camera calibration saved sucessfully')
