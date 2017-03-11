#!/usr/bin/python
## Author: sumanth
## Date: March 08,2017
# main file running the image pipline for lane detection and tracking
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
import imagePipeLine
import cv2
from threshold import threshold, SobelX, HlsGrad, ColorFilt
from perspectiveTransform import warp_and_unwarp_params, warp, unwarp
from laneFinder import find_lines
from imagePipeLine import find_lanes
from line import Line
import pickle

# file names for both input and output
output_file = 'project_output.mp4'
input_file = 'project_video.mp4'

## Choose return type:
# True = 4 image composite
# False = Output only
find_lanes.composite = False

## Calibrate the camera ##
# fetch the camera params
with open('cameraCalibrationParams.pickle', 'rb') as f:
	save_dict = pickle.load(f)
mtx = save_dict['mtx']
dist = save_dict['dist']

find_lanes.mtx = mtx
find_lanes.dist = dist

## Create global variables
LaneLines = Line()

## Load parameters ##
polygon1, polygon2 = warp_and_unwarp_params()
# perspective transform matrices
warp.M = cv2.getPerspectiveTransform(polygon1, polygon2)
unwarp.Minv = cv2.getPerspectiveTransform(polygon2, polygon1)
# lane finder params: number of slices, width of search region, number of pixels
find_lines.num, find_lines.width, find_lines.min = 8, 160, 50
# Sobel and HLS thresholds
SobelX.thresh, HlsGrad.s_thresh = [20, 100], [120, 225]
# Yellow and white thresholds
ColorFilt.yellow, ColorFilt.white = [[215, 255], [140, 255], [0, 160]],[[225, 255], [225, 255], [225, 255]]


#######################
# set the flag to run
run = 'IMAGE'

if run == 'VIDEO':
	# run the pipeline and generate the ouput video
	print("running on video")
	video = VideoFileClip(input_file)
	annotated_video = video.fl_image(lambda img: find_lanes(img, LaneLines))
	annotated_video.write_videofile(output_file, audio=False)
elif run == 'IMAGE':
	print("running on image")
	# testing the pipeline on an image
	im=cv2.imread("test_images/test3.jpg")
	# with visualisation
	combined_img = find_lanes(im, LaneLines, True)
else:
	print("Error set the run flag")
