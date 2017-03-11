
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
import perspectiveTransform
import glob

if __name__ == '__main__':
    img_file = 'test_images/test2.jpg'

    #fetch the camera params
    with open('cameraCalibrationParams.pickle', 'rb') as f:
        save_dict = pickle.load(f)
    mtx = save_dict['mtx']
    dist = save_dict['dist']

    img = mpimg.imread(img_file)
    img = cv2.undistort(img, mtx, dist, None, mtx)

    img, abs_bin, mag_bin, dir_bin, hls_bin = threshold.imageThreshold(img)

    # perspective transform
    src = np.float32(
        [[120, 720],
        [550, 470],
        [700, 470],
        [1160, 720]])

    dst = np.float32(
        [[200,720],
        [200,0],
        [1080,0],
        [1080,720]])

    warped, unwarped, m, m_inv = perspectiveTransform.perspectiveTransform(img, src, dst)

    s=plt.subplot(2, 2, 1)
    plt.imshow(warped, cmap='gray', vmin=0, vmax=1)
    plt.imshow(img)
    s.set_title("wrapped image")

    s=plt.subplot(2, 2, 2)
    plt.imshow(unwarped, cmap='gray', vmin=0, vmax=1)
    plt.imshow(img)
    s.set_title("unwrapped image")

    s=plt.subplot(2, 2, 3)
    plt.imshow(m, cmap='gray', vmin=0, vmax=1)
    plt.imshow(img)
    s.set_title("perspective image")

    s=plt.subplot(2, 2, 4)
    plt.imshow(m_inv, cmap='gray', vmin=0, vmax=1)
    plt.imshow(img)
    s.set_title("inv perspective image")

    plt.tight_layout()
    plt.show()
    plt.savefig('corrected_images/perspectiveTransform_subplot.png')


## testing on test images
images = glob.glob('test_images/straight_lines*.jpg')

count = 0;
for fname in images:
    img = cv2.imread(fname)
    count +=1
    plt.imshow(img)
    plt.savefig('corrected_images/perspective/input' + str(count) + '.png')
    plt.gcf().clear()

    # apply perspective correction
    warped, unwarped, M, Minv = perspectiveTransform.perspectiveTransform(img, src, dst)
    plt.imshow(warped)
    plt.savefig('corrected_images/perspective/output' + str(count) + '.png')
    plt.gcf().clear()
