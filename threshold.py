import glob

from scipy import signal

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle

def threshold(src):
    # Apply both Sobel (X) and HSL (S) thresholds
    # on an image, take union of the two results

    # Sobel value
    sobel_binary = SobelX(src)

    # S value
    s_binary = HlsGrad(src)

    # yellow and white values
    color_binary = ColorFilt(src)

    stack= np.dstack((255*color_binary, 255*sobel_binary, 255*s_binary))

    return stack

def SobelX(src):
    # Apply a Sobel gradient to an image
    # keep only the pixels that lie within the thresholds

    gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    sobel_binary = np.zeros_like(scaled_sobel)
    sobel_binary[(scaled_sobel > SobelX.thresh[0]) & (scaled_sobel <= SobelX.thresh[1])] = 1

    return sobel_binary

def HlsGrad(src):
    # Apply an HLS color transformation on an image
    # keep only the pixels that lie within the thresholds in the S plane

    hls = cv2.cvtColor(src, cv2.COLOR_RGB2HLS)
    s = hls[:,:,2]
    color_binary = np.zeros_like(s)
    color_binary[(s > HlsGrad.s_thresh[0]) & (s <= HlsGrad.s_thresh[1])] = 1

    return color_binary

def ColorFilt(src):
    # Keep only the pixels that lie within the thresholds near yellow and white

    color_binary = np.zeros_like(src[:,:,2])
    color_binary[(((src[:,:,0] > ColorFilt.yellow[0][0]) & (src[:,:,0] < ColorFilt.yellow[0][1]))
                 &((src[:,:,1] > ColorFilt.yellow[1][0]) & (src[:,:,1] < ColorFilt.yellow[1][1]))
                 &((src[:,:,2] > ColorFilt.yellow[2][0]) & (src[:,:,2] < ColorFilt.yellow[2][1])))
                 |(((src[:,:,0] > ColorFilt.white[0][0]) & (src[:,:,0] < ColorFilt.white[0][1]))
                 &((src[:,:,1] > ColorFilt.white[1][0]) & (src[:,:,1] < ColorFilt.white[1][1]))
                 &((src[:,:,2] > ColorFilt.white[2][0]) & (src[:,:,2] < ColorFilt.white[2][1])))] = 1

    return color_binary
