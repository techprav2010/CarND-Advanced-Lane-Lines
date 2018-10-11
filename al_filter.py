import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import glob
import al_consts
from al_calibration import AlCalibration
from al_helper  import AlHelper

KERNEL_SIZE_DEFAULT=5
THRESH_SOBEL_ABS=(50, 255) #(0, 255)
KERNEL_SIZE_SOBEL_ABS=KERNEL_SIZE_DEFAULT


THRESH_SOBEL_MAG=(100, 255) #(0, 255)
KERNEL_SIZE_SOBEL_ABS=KERNEL_SIZE_DEFAULT

THRESH_SOBEL_DIR=(0.7, 1.2) #(0, np.pi / 2)
KERNEL_SIZE_SOBEL_DIR=KERNEL_SIZE_DEFAULT

THRESH_HLS_S = (220, 255)
THRESH_RGB_B = (220, 255)
THRESH_HLS_L= (220, 255)


#a helper class for applying gradient and color filters

class AlFilter:
    ####################################################################

    def __init__(self,  cal_images=al_consts.CALIBRATION_IMAGES  , test_images=al_consts.TEST_IMAGES ):
        self.cal_images = cal_images
        self.test_images = test_images

    def thresh_sobel_abs_x(self, img, orient='x', kernel_size=KERNEL_SIZE_SOBEL_ABS, thresh=THRESH_SOBEL_ABS):
        #BGR2HLS-channel_s,  x or y gradients,  rescale to 8 bit and apply thresh filter
        img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
        img_hls_channel_s = img_hls[:, :, 2]
        if orient == 'x':
            sobel_abs = np.absolute(cv2.Sobel(img_hls_channel_s, cv2.CV_64F, 1, 0, ksize=kernel_size))
        if orient == 'y':
            sobel_abs = np.absolute(cv2.Sobel(img_hls_channel_s, cv2.CV_64F, 0, 1, ksize=kernel_size))
        sobel_scaled = np.uint8(255 * sobel_abs / np.max(sobel_abs))
        img_zero = np.zeros_like(sobel_scaled)
        img_zero[(sobel_scaled >= thresh[0]) & (sobel_scaled <= thresh[1])] = 1
        return img_zero

    def thresh_sobel_mag(self, img, kernel_size=KERNEL_SIZE_SOBEL_ABS, thresh=THRESH_SOBEL_MAG):
        #BGR2HLS-channel_s, square root of x**2 and y**2 gradients,  rescale to 8 bit and  thresh filter
        img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
        img_hls_channel_s = img_hls[:, :, 2]
        sobel_x = cv2.Sobel(img_hls_channel_s, cv2.CV_64F, 1, 0, ksize=kernel_size)
        sobel_y = cv2.Sobel(img_hls_channel_s, cv2.CV_64F, 0, 1, ksize=kernel_size)
        mag_f = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        scale_f = np.max(mag_f) / 255
        grad_mag = (mag_f / scale_f).astype(np.uint8)
        img_zero = np.zeros_like(grad_mag)
        img_zero[(grad_mag >= thresh[0]) & (grad_mag <= thresh[1])] = 1
        return img_zero

    def thresh_sobel_dir(self, img, kernel_size=KERNEL_SIZE_SOBEL_DIR, thresh=THRESH_SOBEL_DIR):
        # BGR2HLS channel 2,  arctan2 of x, y gradients,  apply thresh filter
        img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        img_hls_channel_s = img_hls[:, :, 2]
        sobel_x = cv2.Sobel(img_hls_channel_s, cv2.CV_64F, 1, 0, ksize=kernel_size)
        sobel_y = cv2.Sobel(img_hls_channel_s, cv2.CV_64F, 0, 1, ksize=kernel_size)
        abs_grad_dir = np.arctan2(np.absolute(sobel_y), np.absolute(sobel_x))
        img_zero = np.zeros_like(abs_grad_dir)
        img_zero[(abs_grad_dir >= thresh[0]) & (abs_grad_dir <= thresh[1])] = 1
        return img_zero

    def thresh_hls_l(self, img, thresh=THRESH_HLS_L):
        #apply threshold filter on hls_channel_l
        img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        img_hls_channel_l = img_hls[:, :, 1]
        img_zero = np.zeros_like(img_hls_channel_l)
        img_zero[(img_hls_channel_l >= thresh[0]) & (img_hls_channel_l <= thresh[1])] = 1
        return img_zero

    def thresh_hls_s(self, img, thresh=THRESH_HLS_S):
        # apply threshold filter on hls_channel_s
        img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        img_hls_channel_s = img_hls[:, :, 2]
        img_zero = np.zeros_like(img_hls_channel_s)
        img_zero[(img_hls_channel_s >= thresh[0]) & (img_hls_channel_s <= thresh[1])] = 1
        return img_zero

    def thresh_rgb_b(self, img, thresh=THRESH_RGB_B):
        # apply threshold filter on rgb_channel_b
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb_channel_b = img_rgb[:, :, 2]
        img_zero = np.zeros_like(img_rgb_channel_b)
        img_zero[(img_rgb_channel_b >= thresh[0]) & (img_rgb_channel_b <= thresh[1])] = 1
        return img_zero

    def pipeline(self, img, calibration):
        img = calibration.undistort(img)
        persps = calibration.perspective_transform(img)
        # filter threshold - sobel_mag , sobel_abs_x , _sobel_dir
        persp = persps[0]
        MInv=persps[2]
        img = persp
        bin_img_thresh_sobel_mag = self.thresh_sobel_mag(img)
        bin_img_sobel_abs_x = self.thresh_sobel_abs_x(img, orient='x')
        bin_img_thresh_sobel_dir = self.thresh_sobel_dir(img )

        # filter threshold -  L and B channels
        bin_img_thresh_hls_l = self.thresh_hls_l(img)
        bin_thresh_rgb_b = self.thresh_rgb_b(img)
        img_zero = np.zeros_like(bin_img_sobel_abs_x)
        img_zero[((bin_img_sobel_abs_x == 1)) | ((bin_img_thresh_sobel_mag == 1) & (bin_img_thresh_sobel_dir == 1)) |
                        (bin_img_thresh_hls_l == 1) | (bin_thresh_rgb_b == 1)] = 1
        # img_zero[ (bin_img_thresh_hls_l == 1) | (bin_thresh_rgb_b == 1)] = 1
        return img_zero, MInv, bin_img_thresh_sobel_mag, bin_img_sobel_abs_x, bin_img_thresh_sobel_dir, bin_thresh_rgb_b

    def filter_pipeline(self, img, calibration):
        return self.pipeline(img, calibration)[0]
