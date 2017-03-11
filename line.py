
#!/usr/bin/python
## Author: sumanth
## Date: March 08,2017
# main file for implementing line

import numpy as np

class Line():
    #0 - right
    #1 - left
    def __init__(self):
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = []
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = np.array([[0,0,0],[0,0,0]], dtype='float')
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radii = []
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([[0,0,0],[0,0,0]], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None
        #x location of the first window center form the previous run
        self.center = []
        # conversion of pixels to length
        self.pixels_to_length = []

    def reset(self):
        self.detected = False
        self.recent_xfitted = []
        self.bestx = []
        self.best_fit = np.array([[0,0,0],[0,0,0]], dtype='float')
        self.current_fit = [np.array([False])]
        self.radii = []
        self.line_base_pos = None
        self.diffs = np.array([[0,0,0],[0,0,0]], dtype='float')
        self.allx = None
        self.ally = None
        self.center = []
        self.pixels_to_length = []

    def width_pixels(self):
        #lane width in pixels
        return (self.center[1] - self.center[0])

    def position(self, cols):
        # negative values indicate bias to right line
        # positive values indicate bias to left line
        pos_pixel = (self.center[1]-cols/2)-(cols/2-self.center[0])
        pos_meter = pos_pixel * 3.7 / self.width_pixels()
        return pos_meter

    def draw_lines(self, rows):
        # draw lines using polyfit and EWMA on previous fits
        ploty = np.linspace(0, rows-1, num=rows)
        if self.bestx == []:
            self.bestx = np.zeros((2,rows), dtype = np.float32)
            self.bestx[0] = self.best_fit[0,0]*ploty**2 + self.best_fit[0,1]*ploty + self.best_fit[0,2]
            self.bestx[1] = self.best_fit[1,0]*ploty**2 + self.best_fit[1,1]*ploty + self.best_fit[1,2]
        else:
            tmp = np.zeros((2,rows), dtype = np.float32)
            tmp[0] = self.best_fit[0,0]*ploty**2 + self.best_fit[0,1]*ploty + self.best_fit[0,2]
            tmp[1] = self.best_fit[1,0]*ploty**2 + self.best_fit[1,1]*ploty + self.best_fit[1,2]
            self.bestx[0] = 0.2*tmp[0] + 0.8*self.bestx[0]
            self.bestx[1] = 0.2*tmp[1] + 0.8*self.bestx[1]

        self.radii = np.zeros((2,), dtype = np.float32)

        ym_per_pix = 30/720
        xm_per_pix = 3.7/700
        y_eval = np.max(ploty)
        left_fit_cr = np.polyfit(ploty*ym_per_pix, self.bestx[0]*xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty*ym_per_pix, self.bestx[1]*xm_per_pix, 2)
        # Calculate the new radii of curvature
        self.radii[0] = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        self.radii[1] = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
