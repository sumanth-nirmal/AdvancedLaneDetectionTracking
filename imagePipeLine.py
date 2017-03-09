#!/usr/bin/python
## Author: sumanth
## Date: March 08,2017
# image pipline for lane tracking

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import threshold
import perspectiveTransform
import laneFinder

# fetch the camera params
with open('cameraCalibrationParams.pickle', 'rb') as f:
	save_dict = pickle.load(f)
mtx = save_dict['mtx']
dist = save_dict['dist']

old_l_coff = None
old_r_coff = None

# image pipe line function
def imagePipeLine(file, filepath=False):

    global old_l_coff
    global old_r_coff

    if filepath == True:
        # Read in image
        raw_image = cv2.imread(file)
    else:
        raw_image = file

    # Parameters
    imshape = raw_image.shape

    # Coordinates of quadrangle vertices to get bird eye view
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

    # step - 1 get the perspective transform of a image
    warped, unwarped, M, Minv = perspectiveTransform.perspectiveTransform(raw_image, src, dst)
    #M = cv2.getPerspectiveTransform(src, dst)
    #Minv = cv2.getPerspectiveTransform(dst, src)

    height = raw_image.shape[0]
    offset = 50
    offset_height = height - offset
    half_frame = raw_image.shape[1] // 2
    steps = 6
    pixels_per_step = offset_height / steps
    window_radius = 200
    medianfilt_kernel_size = 51

    blank_canvas = np.zeros((720, 1280))
    colour_canvas = cv2.cvtColor(blank_canvas.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    # Apply distortion correction to raw_image image
    image = cv2.undistort(raw_image, mtx, dist, None, mtx)

    ## Option I
    combined, abs_bin, mag_bin, dir_bin, hls_bin = threshold.imageThreshold(image)

    ## Option II

    have_fit = False
    curvature_checked = False

    xgrad_thresh_temp = (40,100)
    s_thresh_temp=(150,255)

    while have_fit == False:
        combined_binary = threshold.applyThresholdColorHLS(image, xgrad_thresh=xgrad_thresh_temp, s_thresh=s_thresh_temp)
    #    plt.imshow(combined_binary, cmap="gray")

        # Plotting thresholded images
        """
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        ax1.set_title('Option 1')
        ax1.imshow(combined, cmap="gray")

        ax2.set_title('Option 2: Combined S channel and gradient thresholds')
        ax2.imshow(combined_binary, cmap='gray')
        """

        # Warp onto birds-eye-view
        # Previous region-of-interest mask's function is absorbed by the warp
        warped = cv2.warpPerspective(combined_binary, M, (imshape[1], imshape[0]), flags=cv2.INTER_LINEAR)
        # plt.imshow(warped, cmap="gray")

        # Histogram and get pixels in window

        leftx, lefty, rightx, righty = laneFinder.histogramPixels(warped, horizontal_offset=40)

        plt.imshow(warped, cmap="gray")

        if len(leftx) > 1 and len(rightx) > 1:
            have_fit = True
        xgrad_thresh_temp = (xgrad_thresh_temp[0] - 2, xgrad_thresh_temp[1] + 2)
        s_thresh_temp = (s_thresh_temp[0] - 2, s_thresh_temp[1] + 2)

    left_fit, left_coeffs = laneFinder.fit2OrderPoly(lefty, leftx, return_coeffs=True)
    print("Left coeffs:", left_coeffs)
    print("righty[0]: ,", righty[0], ", rightx[0]: ", rightx[0])
    right_fit, right_coeffs = laneFinder.fit2OrderPoly(righty, rightx, return_coeffs=True)
    print("Right coeffs: ", right_coeffs)

    # Plot data
    """
    plt.plot(left_fit, lefty, color='green', linewidth=3)
    plt.plot(right_fit, righty, color='green', linewidth=3)
    plt.imshow(warped, cmap="gray")
    """

    # Determine curvature of the lane
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = 500
    left_curverad = np.absolute(((1 + (2 * left_coeffs[0] * y_eval + left_coeffs[1])**2) ** 1.5) \
                    /(2 * left_coeffs[0]))
    right_curverad = np.absolute(((1 + (2 * right_coeffs[0] * y_eval + right_coeffs[1]) ** 2) ** 1.5) \
                     /(2 * right_coeffs[0]))
    print("Left lane curve radius: ", left_curverad)
    print("Right lane curve radius: ", right_curverad)
    curvature = (left_curverad + right_curverad) / 2
    min_curverad = min(left_curverad, right_curverad)

    # TODO: if plausible parallel, continue. Else don't make `curvature_checked` = True
    if not laneFinder.isCurvature(left_curverad, right_curverad) or \
        not laneFinder.isTraces(left_coeffs, right_coeffs, old_l_coff, old_r_coff):
            if old_l_coff is not None and old_r_coff is not None:
                left_coeffs = old_l_coff
                right_coeffs = old_r_coff

    old_l_coff = left_coeffs
    old_r_coff = right_coeffs

    # Det vehicle position wrt centre
    centre = laneFinder.getCenter(719, left_coeffs, right_coeffs)

    ## 7. Warp the detected lane boundaries back onto the original image.

    print("Left coeffs: ", left_coeffs)
    print("Right fit: ", right_coeffs)
    polyfit_left = laneFinder.drawPoly(blank_canvas, laneFinder.lanePoly, left_coeffs, 30)
    polyfit_drawn = laneFinder.drawPoly(polyfit_left, laneFinder.lanePoly, right_coeffs, 30)
#    plt.imshow(polyfit_drawn, cmap="gray")
#    plt.imshow(warped)

    # Convert to colour and highlight lane line area
    trace = colour_canvas
    trace[polyfit_drawn > 1] = [0,0,255]
    print("polyfit shape: ", polyfit_drawn.shape)
    area = laneFinder.getLaneArea(blank_canvas, left_coeffs, right_coeffs)
    trace[area == 1] = [0,255,0]
    # plt.imshow(trace)
    lane_lines = cv2.warpPerspective(trace, Minv, (imshape[1], imshape[0]), flags=cv2.INTER_LINEAR)
    # plt.imshow(trace)

    combined_img = cv2.add(lane_lines, image)
    laneFinder.addInfoToImage(combined_img, curvature=curvature,
                         vehicle_position=centre,
                         min_curvature=min_curverad,
                         left_coeffs=left_coeffs,
                         right_coeffs=right_coeffs)
    plt.imshow(combined_img)
    return combined_img
