import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import glob


#put consultants  here
CALIBRATION_IMAGES = glob.glob('camera_cal/*.jpg')
TEST_IMAGES = glob.glob('test_images/*.jpg')