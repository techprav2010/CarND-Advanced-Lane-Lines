import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import glob
import os
import al_consts
from al_helper import AlHelper
from al_calibration import AlCalibration


WINDOW_COUNT =60
MARGIN = 80
MIN_PIXEL = 40

#class to do => poly fit, sliding window algorithm, draw curve, track previous lane fit, draw vehicle position and lane curvature information
class AlAdvLane:

    def __init__(self ):
        self.al_helper = AlHelper()
        self.calibration = AlCalibration(9, 6, cal_images=al_consts.CALIBRATION_IMAGES, test_images=al_consts.TEST_IMAGES)
        self.calibration.calibrate_camera()

    def window_sliding_rects(self, img):
        #sliding window algorithm

        # lane pixel indices - left and right, rectangle_data
        left_lane_inds, right_lane_inds, rectangle_data = [], [], []

        # windows
        window_count = WINDOW_COUNT
        margin = MARGIN
        min_pixel = MIN_PIXEL
        window_height = np.int(img.shape[0] / window_count)

        # find all non zero x,y
        non_zero = img.nonzero()
        non_zero_x = np.array(non_zero[1])
        non_zero_y = np.array(non_zero[0])
        
        # histogram -  left and right
        histogram = np.sum(img[img.shape[0] // 2:, :], axis=0)
        mid_point = np.int(histogram.shape[0] // 2)
        quarter_point = np.int(mid_point // 2)
        # points : left and right histogram
        left_x_base  = np.argmax(histogram[quarter_point:mid_point]) + quarter_point
        right_x_base = np.argmax(histogram[mid_point:(mid_point + quarter_point)]) + mid_point
        left_x_current = left_x_base # loop var
        right_x_current = right_x_base # loop var

        for window in range(window_count):
            # window x, y boundaries - right and left
            window_y_low = img.shape[0] - (window + 1) * window_height
            window_y_high = img.shape[0] - window * window_height
            
            window_x_left_low = left_x_current - margin
            window_x_left_high = left_x_current + margin
            window_x_right_low = right_x_current - margin
            window_x_right_high = right_x_current + margin

            #rectangle co-ordinate
            rectangle_data.append((window_y_low, window_y_high, window_x_left_low, window_x_left_high, window_x_right_low, window_x_right_high))
            
            # find x and y wrt to window
            good_left_inds = ((non_zero_y >= window_y_low) & (non_zero_y < window_y_high) & (non_zero_x >= window_x_left_low) &
                              (non_zero_x < window_x_left_high)).nonzero()[0]
            good_right_inds = ((non_zero_y >= window_y_low) & (non_zero_y < window_y_high) & (non_zero_x >= window_x_right_low) &
                               (non_zero_x < window_x_right_high)).nonzero()[0]
            # keep these points
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            # mean - left and right in window
            if len(good_left_inds) > min_pixel:
                left_x_current = np.int(np.mean(non_zero_x[good_left_inds]))
            if len(good_right_inds) > min_pixel:
                right_x_current = np.int(np.mean(non_zero_x[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        left_x = non_zero_x[left_lane_inds]
        left_y = non_zero_y[left_lane_inds]
        right_x = non_zero_x[right_lane_inds]
        right_y = non_zero_y[right_lane_inds]

        left_fit, right_fit = (None, None)
        # Fit a second order polynomial to each
        if len(left_x) != 0:
            left_fit = np.polyfit(left_y, left_x, 2)
        if len(right_x) != 0:
            right_fit = np.polyfit(right_y, right_x, 2)


        return left_fit, right_fit, left_lane_inds, right_lane_inds, rectangle_data, histogram

    def poly_fit_prev(self, img, left_fit, right_fit):
        #Poly fit
        # lane pixel indices - left and right, rectangle_data
        margin = 80

        # find all non zero x,y
        non_zero = img.nonzero()
        non_zero_x = np.array(non_zero[1])
        non_zero_y = np.array(non_zero[0])

        left_fit_prev = (left_fit[0] * (non_zero_y ** 2) + left_fit[1] * non_zero_y + left_fit[2]) * 1.0
        right_fit_prev = (right_fit[0] * (non_zero_y ** 2) + right_fit[1] * non_zero_y + right_fit[2]) * 1.0

        # find x and y wrt to window

        left_lane_inds = ((non_zero_x > (left_fit[0] * (non_zero_y ** 2) + left_fit[1] * non_zero_y + left_fit[2] - margin))
                          & (non_zero_x < (left_fit[0] * (non_zero_y ** 2) + left_fit[1] * non_zero_y + left_fit[2] + margin)))

        right_lane_inds = ((non_zero_x > (right_fit[0] * (non_zero_y ** 2) + right_fit[1] * non_zero_y + right_fit[2] - margin))
                           & (non_zero_x < (right_fit[0] * (non_zero_y ** 2) + right_fit[1] * non_zero_y + right_fit[2] + margin)))

        left_x = non_zero_x[left_lane_inds]
        left_y = non_zero_y[left_lane_inds]
        right_x = non_zero_x[right_lane_inds]
        right_y = non_zero_y[right_lane_inds]

        left_fit_new, right_fit_new = (None, None)

        if len(left_x) != 0:
            left_fit_new = np.polyfit(left_y, left_x, 2)
        if len(right_x) != 0:
            right_fit_new = np.polyfit(right_y, right_x, 2)

        return left_fit_new, right_fit_new, left_lane_inds, right_lane_inds


    def curve_position(self, img, left_fit, right_fit, left_lane_inds, right_lane_inds):
        #find lane curve positon
        left_curve_rad, right_curve_rad, dist_center = (0, 0, 0)

        non_zero = img.nonzero()
        non_zero_y = np.array(non_zero[0])
        non_zero_x = np.array(non_zero[1])

        # non zero - left and right
        left_x = non_zero_x[left_lane_inds]
        left_y = non_zero_y[left_lane_inds]
        right_x = non_zero_x[right_lane_inds]
        right_y = non_zero_y[right_lane_inds]

        # pixels to meters : { y_m_per_pix = 10ft = 3.048m , x_m_per_pix = 12ft = 3.7m}
        y_m_per_pix = 3.048 / 100
        x_m_per_pix = 3.7 / 378

        #plot ys
        height = img.shape[0]
        plot_y = np.linspace(0, height - 1, height)

        # radius of curvature: maximum y-value
        max_y = np.max(plot_y)

        if len(left_x) != 0 and len(right_x) != 0:
            # world space : polynomials
            left_fit_cr = np.polyfit(left_y * y_m_per_pix, left_x * x_m_per_pix, 2)
            right_fit_cr = np.polyfit(right_y * y_m_per_pix, right_x * x_m_per_pix, 2)
            # radii of curvature : in meters
            left_curve_rad  = ((1 + ( 2 * left_fit_cr[0]  * max_y * y_m_per_pix + left_fit_cr[1]) ** 2) ** 1.5)  / np.absolute(  2 * left_fit_cr[0])
            right_curve_rad = ((1 + ( 2 * right_fit_cr[0] * max_y * y_m_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(  2 * right_fit_cr[0])

        # distance - center : image x midpoint and mean of left_fit and right_fit intercepts
        if right_fit is not None and left_fit is not None:
            pos_car = img.shape[1] / 2
            left_fit_x_int = left_fit[0] * height ** 2 + left_fit[1] * height + left_fit[2]
            right_fit_x_int = right_fit[0] * height ** 2 + right_fit[1] * height + right_fit[2]
            pos_car_center = (right_fit_x_int + left_fit_x_int) / 2
            dist_center = (pos_car - pos_car_center) * x_m_per_pix
        return left_curve_rad, right_curve_rad, dist_center

    def fill_lane(self, img, img_filter, left_fit, right_fit, Minv):
        # take the filter => unwrap , draw
        new_img = np.copy(img)
        if left_fit is None or right_fit is None:
            return img

        warp_zero = np.zeros_like(img_filter).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        height, weidth = img_filter.shape

        plot_y = np.linspace(0, height - 1, num=height)
        left_fit_x = left_fit[0] * plot_y ** 2 + left_fit[1] * plot_y + left_fit[2]
        right_fit_x = right_fit[0] * plot_y ** 2 + right_fit[1] * plot_y + right_fit[2]

        left_pts = np.array([np.transpose(np.vstack([left_fit_x, plot_y]))])
        right_pts = np.array([np.flipud(np.transpose(np.vstack([right_fit_x, plot_y])))])
        pts = np.hstack((left_pts, right_pts))

        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        cv2.polylines(color_warp, np.int32([left_pts]), isClosed=False, color=(255, 0, 255), thickness=15)
        cv2.polylines(color_warp, np.int32([right_pts]), isClosed=False, color=(0, 255, 255), thickness=15)

        newwarp = cv2.warpPerspective(color_warp, Minv, (weidth, height))
        result = cv2.addWeighted(new_img, 1, newwarp, 0.5, 0)
        return result

    def write_curve_info(self, original_img, curv_rad, center_dist):
        #write data on image
        new_img = np.copy(original_img)
        h = new_img.shape[0]
        font = cv2.FONT_HERSHEY_DUPLEX
        Curvature = 'Curvature of the lane : ' + '{:04.2f}'.format(curv_rad) + ' m'
        cv2.putText(new_img, Curvature, (40, 70), font, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
        direction = ''
        if center_dist > 0:
            direction = 'right'
        elif center_dist < 0:
            direction = 'left'
        abs_center_dist = abs(center_dist)
        vehicle_position = 'Vehicle position : {:04.3f}'.format(abs_center_dist) + ' m ' + direction + ' from center'
        cv2.putText(new_img, vehicle_position, (40, 120), font, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
        return new_img



    ###############################################################################
    # helper method to draw on notebook
    ###############################################################################
    # helper method to draw on notebook
    def draw_window_sliding(self, img):
        # windows
        window_count = WINDOW_COUNT
        margin = MARGIN
        min_pixel = MIN_PIXEL
        window_height = np.int(img.shape[0] / window_count)

        # find all non zero x,y
        non_zero = img.nonzero()
        non_zero_x = np.array(non_zero[1])
        non_zero_y = np.array(non_zero[0])

        left_fit, right_fit, left_lane_inds, right_lane_inds, rectangle_data, histogram = self.window_sliding_rects(img)

        ###############################################################################
        # create img_windows_1
        img_windows_1 = np.uint8(np.dstack((img, img, img)) * 255)
        for rect in rectangle_data:
            cv2.rectangle(img_windows_1, (rect[2], rect[0]), (rect[3], rect[1]), (0, 255, 0), 2)
            cv2.rectangle(img_windows_1, (rect[4], rect[0]), (rect[5], rect[1]), (0, 255, 0), 2)

        img_windows_1[non_zero_y[left_lane_inds], non_zero_x[left_lane_inds]] = [255, 0, 0]
        img_windows_1[non_zero_y[right_lane_inds], non_zero_x[right_lane_inds]] = [100, 200, 255]

        plot_y = np.linspace(0, img.shape[0] - 1, img.shape[0])
        left_fit_x = left_fit[0] * plot_y ** 2 + left_fit[1] * plot_y + left_fit[2]
        right_fit_x = right_fit[0] * plot_y ** 2 + right_fit[1] * plot_y + right_fit[2]


        left_pts = np.vstack((left_fit_x, plot_y)).astype(np.int32).T
        right_pts = np.vstack((right_fit_x, plot_y)).astype(np.int32).T

        # draw lanes on img_windows_1
        cv2.polylines(img_windows_1, [left_pts], False, (0, 255, 0), 10)
        cv2.polylines(img_windows_1, [right_pts], False, (0, 255, 0), 10)

        ###############################################################################
        # create img_windows_4
        img_windows_2 = np.dstack((img, img, img)) * 255
        img_windows_3 = np.zeros_like(img_windows_2)

        # Color line pixels
        img_windows_2[non_zero_y[left_lane_inds], non_zero_x[left_lane_inds]] = [255, 0, 0]
        img_windows_2[non_zero_y[right_lane_inds], non_zero_x[right_lane_inds]] = [0, 0, 255]

        # x and y points for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fit_x - margin, plot_y]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fit_x + margin,  plot_y])))])

        # x and y points for cv2.fillPoly()
        right_line_window1 = np.array([np.transpose(np.vstack([right_fit_x - margin, plot_y]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fit_x + margin,    plot_y])))])

        #stack
        img_windows_2a = np.hstack((left_line_window1, left_line_window2))
        img_windows_2b = np.hstack((right_line_window1, right_line_window2))

        # draw lane onto image
        cv2.fillPoly(img_windows_3, np.int_([img_windows_2a]), (0, 255, 0))
        cv2.fillPoly(img_windows_3, np.int_([img_windows_2b]), (0, 255, 0))

        img_windows_4 = cv2.addWeighted(img_windows_2, 1, img_windows_3, 0.3, 0)

        mg_histogram = self._histo_img(img)
        # img_curve = self._image_draw_lanes_2(img, left_fit_x, right_fit_x , plot_y)
        return left_fit, right_fit, left_lane_inds, right_lane_inds, rectangle_data, histogram, img_windows_1, img_windows_4 , mg_histogram #, img_curve



    ###############################################################################
    # helper method to draw on notebook
    def draw_curve(self, img_filter, img):
        margin = MARGIN #50
        non_zero = img_filter.nonzero()
        non_zero_y = np.array(non_zero[0])
        non_zero_x = np.array(non_zero[1])

        left_fit_1, right_fit_1, left_lane_inds_1, right_lane_inds_1, visualization_data , histogram = self.window_sliding_rects(img_filter)
        left_fit_2, right_fit_2, left_lane_inds_2, right_lane_inds_2 = self.poly_fit_prev(img_filter, left_fit_1, right_fit_1)

        plot_y = np.linspace(0, img_filter.shape[0] - 1, img_filter.shape[0])

        left_fit_x_1 = left_fit_1[0] * plot_y ** 2 + left_fit_1[1] * plot_y + left_fit_1[2]
        right_fit_x_1 = right_fit_1[0] * plot_y ** 2 + right_fit_1[1] * plot_y + right_fit_1[2]
        left_fit_x_2 = left_fit_2[0] * plot_y ** 2 + left_fit_2[1] * plot_y + left_fit_2[2]
        right_fit_x_2 = right_fit_2[0] * plot_y ** 2 + right_fit_2[1] * plot_y + right_fit_2[2]

        img_windows_1 = np.uint8(np.dstack((img_filter, img_filter, img_filter)) * 255)
        window_img_2 = np.zeros_like(img_windows_1)
        img_windows_1[non_zero_y[left_lane_inds_2], non_zero_x[left_lane_inds_2]] = [255, 0, 0]
        img_windows_1[non_zero_y[right_lane_inds_2], non_zero_x[right_lane_inds_2]] = [0, 0, 255]

        left_line_window_1 = np.array([np.transpose(np.vstack([left_fit_x_1 - margin, plot_y]))])
        left_line_window_2 = np.array([np.flipud(np.transpose(np.vstack([left_fit_x_1 + margin, plot_y])))])
        left_line_pts = np.hstack((left_line_window_1, left_line_window_2))

        right_line_window_1 = np.array([np.transpose(np.vstack([right_fit_x_1 - margin, plot_y]))])
        right_line_window_2 = np.array([np.flipud(np.transpose(np.vstack([right_fit_x_1 + margin, plot_y])))])
        right_line_pts = np.hstack((right_line_window_1, right_line_window_2))

        cv2.fillPoly(window_img_2, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img_2, np.int_([right_line_pts]), (0, 255, 0))
        window_img_3 = cv2.addWeighted(img_windows_1, 1, window_img_2, 0.3, 0)

        window_img_4 = self._curve_img(window_img_3, left_fit_x_2, right_fit_x_2, plot_y  )
        # plt.plot(left_fit_x_2, plot_y, color='yellow')
        # plt.plot(right_fit_x_2, plot_y, color='yellow')

        return window_img_4

    ###############################################################################
    # helper method to draw on notebook
    def draw_line(self, img, original, Minv):

        left_fit, right_fit, left_lane_inds, right_lane_inds, visualization_data, histogram  = self.window_sliding_rects(img)
        left_fit2, right_fit2, left_lane_inds2, right_lane_inds2 = self.poly_fit_prev(img, left_fit, right_fit)
        rad_l, rad_r, d_center = self.curve_position(img, left_fit, right_fit, left_lane_inds2, right_lane_inds2)

        result = self.fill_lane(original, img, left_fit, right_fit, Minv)
        result = self.write_curve_info(result, (rad_l + rad_r) / 2, d_center)
        return result

    ###############################################################################
    # helper method to draw on notebook
    def pipeline_image_debug(self, img, left_lane, right_lane , calibration):
        from al_filter import AlFilter
        al_filter=AlFilter()
        new_img = np.copy(img)
        img_filtered, Minv, *b = al_filter.pipeline(img, calibration)
        if not left_lane.detected or not right_lane.detected:
            left_fit, right_fit, left_lane_inds, right_lane_inds, visualization_data, histogram = self.window_sliding_rects(img_filtered)
        else:
            left_fit, right_fit, left_lane_inds, right_lane_inds = self.polyfit_prev_fit(img_filtered, left_lane.best_fit, right_lane.best_fit)

        left_lane.add_fit(left_fit, left_lane_inds)
        right_lane.add_fit(right_fit, right_lane_inds)
        img_windows_1 = self.fill_lane(new_img, img_filtered, left_fit, right_fit, Minv)
        rad_left, rad_right, d_center = self.curve_position(img_filtered, left_fit, right_fit, left_lane_inds, right_lane_inds)
        img_windows_2 = self.write_curve_info(img_windows_1, (rad_left + rad_right) / 2, d_center)
        return img_windows_2


    ###############################################################################
    #internalyy used helper methods
    def _curve_img(self, img, left_fit_x_2, right_fit_x_2, plot_y  ):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        plt.imshow(img)
        plt.xlabel('poly fit using previous fit', fontsize=15)
        plt.plot(left_fit_x_2, plot_y, color='yellow')
        plt.plot(right_fit_x_2, plot_y, color='yellow')
        # plt.xlim(0, 1280)
        # plt.ylim(720, 0)
        plt.xticks([])
        plt.yticks([])
        fig.savefig('output_images/curve_img.png')
        plt.close(fig)
        return self._load_image("output_images/curve_img.png")

    def _histo_img(self, img):
        histogram = np.sum(img[img.shape[0] // 2:, :], axis=0)
        fig, ax = plt.subplots(nrows=1, ncols=1)
        plt.plot(histogram)
        fig.savefig('output_images/histgram.png')
        plt.close(fig)
        return self._load_image("output_images/histgram.png")

    def _load_image(self, cal_img):
        return mpimg.imread(cal_img)

