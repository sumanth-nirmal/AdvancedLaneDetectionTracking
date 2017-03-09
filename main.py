#!/usr/bin/python
## Author: sumanth
## Date: March 08,2017
# main file running the image pipline for lane detection and tracking

from moviepy.editor import VideoFileClip
import imagePipeLine

# file names for both input and output
output_file = 'project_challenge_output_colour.mp4'
input_file = 'challenge_video.mp4'


# run the pipeline and generate the ouput video
video = VideoFileClip(input_file)
annotated_video = video.fl_image(imagePipeLine.imagePipeLine)
annotated_video.write_videofile(output_file, audio=False)


# testing the pipeline on an image
combined_img = imagePipeLine("test_images/test1.jpg", filepath=True)
plt.imshow(combined_img)
plt.savefig('corrected_images/imageAfterPipeLine.png')
