#!/usr/bin/python
## Author: sumanth
## Date: March 08,2017
# methods for perspective transformation

import numpy as np
import cv2

def perspectiveTransform(img):
	img_size = (img.shape[1], img.shape[0])

    #Coordinates of quadrangle vertices
    # perspective transform to rectify binary image ("birds-eye view")
	src = np.float32(
		[[200, 720],
		[1100, 720],
		[595, 450],
		[685, 450]])
	dst = np.float32(
		[[300, 720],
		[980, 720],
		[300, 0],
		[980, 0]])

	m = cv2.getPerspectiveTransform(src, dst)
	m_inv = cv2.getPerspectiveTransform(dst, src)

	warped = cv2.warpPerspective(img, m, img_size, flags=cv2.INTER_LINEAR)
	unwarped = cv2.warpPerspective(warped, m_inv, (warped.shape[1], warped.shape[0]), flags=cv2.INTER_LINEAR)

	return warped, unwarped, m, m_inv
