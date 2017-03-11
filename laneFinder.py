import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import signal

def paint_lane(src, LaneLines):
    # Given an image and a polynomial best fit for each edge
    # superimpose the lane over the image
    rows = src.shape[0]
    dst = np.zeros_like(src).astype(np.uint8)
    LaneLines.draw_lines(rows)
    ploty = np.linspace(0, rows-1, num=rows)

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([LaneLines.bestx[0], ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([LaneLines.bestx[1], ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(dst, np.int_([pts]), (0,255, 0))

    return dst

def find_lines(src, LaneLines):
    # Divide the images in N horizontal slices and find the lane
    # lines in each slice

    src = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)


    rows, cols = src.shape[0], src.shape[1]

    # Basic geometry of the figure, slice up into sections
    slice_height = np.uint32(np.trunc(rows / find_lines.num))
    slice_width = np.uint32(np.trunc(cols / 2))
    mid_height = np.uint32(np.trunc(rows / 2))

    # Find the lane pixels
    nonzero_row = np.array(src.nonzero()[0])
    nonzero_col = np.array(src.nonzero()[1])

    # Sum up along the vertical direction, find a starting point to search
    hist = np.sum(src[mid_height:,:], axis=0)

    left_indices_all = []
    right_indices_all = []
    if LaneLines.center == []:
        left_center = np.argmax(hist[:slice_width]) # horizontal center of the left search rectangle
        right_center = slice_width+np.argmax(hist[slice_width:]) # horizontal center of the right search rectangle
    else:
        window_left = LaneLines.center[0]-np.uint32(find_lines.width/2)
        window_right = LaneLines.center[0]+np.uint32(find_lines.width/2)
        left_center = window_left+np.argmax(hist[window_left:window_right])
        window_left = LaneLines.center[1]-np.uint32(find_lines.width/2)
        window_right = LaneLines.center[1]+np.uint32(find_lines.width/2)
        right_center = window_left+np.argmax(hist[window_left:window_right])
    LaneLines.center = [np.uint32(left_center), np.uint32(right_center)]

    dst = np.dstack((src, src, src))*255

    for idx in range(find_lines.num):
        ## rectangle borders

        # top and bottom border of both left and right rectangles
        top = rows - np.uint32((idx+1)*slice_height)
        bottom = rows - np.uint32((idx)*slice_height)

        # left and right border of left rectangle
        if left_center <= (find_lines.width/2):
            left_left = 0
        else:
            left_left = left_center-np.uint32(find_lines.width/2)

        left_right = left_center+np.uint32(find_lines.width/2)

        # left and right border of right rectangle
        right_left = right_center-np.uint32(find_lines.width/2)
        if right_center >= cols-np.uint32(find_lines.width/2):
            right_right = cols
        else:
            right_right = right_center+np.uint32(find_lines.width/2)

        # draw rectangles
        cv2.rectangle(dst, (left_left,bottom),(left_right,top),(255,165,0), 2) # orange
        cv2.rectangle(dst, (right_left,bottom),(right_right,top),(0,165,1255), 2)

        # search rectangles for nonzero points
        left_indices = ((nonzero_row >= top) & (nonzero_row < bottom)
                          & (nonzero_col >= left_left) & (nonzero_col < left_right)).nonzero()[0]
        right_indices = ((nonzero_row >= top) & (nonzero_row < bottom)
                          & (nonzero_col >= right_left) & (nonzero_col < right_right)).nonzero()[0]

        # Update search reach region for the next rectangle
        # if many pixels suggest to do so
        if len(left_indices) >= find_lines.min:
            left_center = np.uint32(np.mean(nonzero_col[left_indices]))

        if len(right_indices) >= find_lines.min:
            right_center = np.uint32(np.mean(nonzero_col[right_indices]))
        left_indices_all.append(left_indices)
        right_indices_all.append(right_indices)

    left_indices_all = np.concatenate(left_indices_all)
    right_indices_all = np.concatenate(right_indices_all)

    LaneLines.best_fit[0] = np.polyfit(nonzero_row[left_indices_all],nonzero_col[left_indices_all],2)
    LaneLines.best_fit[1] = np.polyfit(nonzero_row[right_indices_all],nonzero_col[right_indices_all],2)

    dst[nonzero_row[left_indices_all], nonzero_col[left_indices_all]] = [255, 0, 0]
    dst[nonzero_row[right_indices_all], nonzero_col[right_indices_all]] = [0, 0, 255]

    return dst

def composite_image(img1, img2, img3, img4):
    # show 4 images on the same plot for tuning analysis
    dst = np.zeros_like(img1, dtype = np.uint8)
    dst[0:360,0:640] = cv2.resize(img1, (640,360))
    dst[0:360,640:1281] = cv2.resize(img2, (640,360))
    dst[360:721,0:640] = cv2.resize(img3, (640,360))
    dst[360:721,640:1281] = cv2.resize(img4, (640,360))

    return dst

def annotate(img, LaneLines):
    # annotate image with line information
    rows, cols = img.shape[0], img.shape[1]
    pos = LaneLines.position(cols)
    lrad = LaneLines.radii[0] #left lane radius
    rrad = LaneLines.radii[1] #right lane radius
    font = cv2.FONT_HERSHEY_SIMPLEX
    dst = img
    if pos > 0:
        cv2.putText(dst,'Position = {:1.2}m left'.format(pos), (np.int(cols/2)-100,50), font, 1,(255,255,255),2)
    else:
        cv2.putText(dst,'Position = {:1.2}m right'.format(-pos), (np.int(cols/2)-100,50), font, 1,(255,255,255),2)

    cv2.putText(dst,'Left curve radius = {:.0f}m'.format(lrad), (np.int(cols/2)-100,100), font, 1,(255,255,255),2)
    cv2.putText(dst,'Right curve radius = {:.0f}m'.format(rrad), (np.int(cols/2)-100,150), font, 1,(255,255,255),2)

    return dst
