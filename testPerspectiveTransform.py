
#!/usr/bin/python
## Author: sumanth
## Date: March 08,2017
# test script for perspective transformation

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import cv2
import threshold
import perspectiveTransform

if __name__ == '__main__':
    img_file = 'test_images/test5.jpg'

    #fetch the camera params
    with open('cameraCalibrationParams.pickle', 'rb') as f:
        save_dict = pickle.load(f)
    mtx = save_dict['mtx']
    dist = save_dict['dist']

    img = mpimg.imread(img_file)
    img = cv2.undistort(img, mtx, dist, None, mtx)

    img, abs_bin, mag_bin, dir_bin, hls_bin = threshold.imageThreshold(img)
    warped, unwarped, m, m_inv = perspectiveTransform.perspectiveTransform(img)

    s=plt.subplot(1, 2, 1)
    plt.imshow(warped, cmap='gray', vmin=0, vmax=1)
    plt.imshow(img)
    s.set_title("wrapped image")

    s=plt.subplot(1, 2, 2)
    plt.imshow(unwarped, cmap='gray', vmin=0, vmax=1)
    plt.imshow(img)
    s.set_title("unwrapped image")

    plt.tight_layout()
    plt.show()
    plt.savefig('corrected_images/perspectiveTransform_subplot.png')
