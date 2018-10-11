import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import glob
import al_consts


#Whelper class to calibrate camera and undo distortion from images

class AlCalibration:

    def __init__(self, nx=9, ny=6 , cal_images=al_consts.CALIBRATION_IMAGES  , test_images=al_consts.TEST_IMAGES ):
        self.cal_images = cal_images
        self.test_images = test_images
        self.nx, self.ny = 9, 6
        self.objp = np.zeros((nx * ny, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    def calibrate_camera(self):
        objpoints = []  # 3D points
        imgpoints = []  # 2D points

        objp = np.zeros((self.nx * self.ny, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.nx, 0:self.ny].T.reshape(-1, 2)

        for file in self.cal_images:
            img = cv2.imread(file)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (self.nx, self.ny), None)
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)
        #matrix, distortion coefficients, rotation and translation vector
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        self.mtx = mtx
        self.dist = dist
        save_dict = {'matrix': mtx, 'distortion_coef': dist}
        with open('calibrate_camera.p', 'wb') as f:
            pickle.dump(save_dict, f)
        return self.mtx, self.dist

    def draw(self, img, draw=False):
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #  chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            cv2.cornerSubPix(gray, corners, (13, 13), (-1, -1), criteria)
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (9, 6), corners, ret)
            if draw:
                cv2.imshow('img', img)
                cv2.waitKey(500)
        return img

    def load_calibrate_camera(self):
        #load saved data from pickle
        with open('calibrate_camera.p', 'rb') as f:
            save_dict = pickle.load(f)
        self.mtx = save_dict['matrix']
        self.dist = save_dict['distortion_coef']
        return self.mtx, self.dist

    def undistort(self,img):
        if self.mtx  is None:
            self.calibrate_camera()
        dst = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        return dst

    def _src_dest_frame(self, img):
        #return points for warping one image into another.
        src = np.float32([[190, 700], [1110, 700], [720, 470], [570, 470]])
        bottom_left = src[0][0] + 100, src[0][1]
        bottom_right = src[1][0] - 200, src[1][1]
        top_left = src[3][0] - 250, 1
        top_right = src[2][0] + 200, 1
        dst = np.float32([bottom_left, bottom_right, top_right, top_left])
        return src, dst


    def perspective_transform(self, img):
        #do perspective_ transformation using src, dst -> from dashboard view to birds eye view.
        w = img.shape[1]
        h = img.shape[0]
        img_size = (w, h)
        src, dst = self._src_dest_frame(img)
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
        return warped, M, Minv

    def perspective_transform_with_poi(self, img, original_image):
        warped, M, Minv = self.perspective_transform(img)
        #original idea was to draw poi
        preview = np.copy(original_image)
        return preview, warped, M, Minv


