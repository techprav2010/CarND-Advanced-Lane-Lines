import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import glob
import al_consts


#Simnple class to holds lane status for video pipeline
class AlLane():
    def __init__(self):
        self.detected = False # lane detected ?
        self.diffs = np.array([0, 0, 0], dtype='float') # diff coefficients -  between last and new fits
        self.best_fit = None # averaged fit poly-coeff
        self.current_fit = []  # recent fit poly-coeff
    def add(self, fit, inds):
        if fit is not None:
            self.detected = True
            self.current_fit.append(fit)
            if len(self.current_fit) > 5:
                self.current_fit = self.current_fit[len(self.current_fit)-5:]
            self.best_fit = np.average(self.current_fit, axis=0)
        else:
            self.detected = False
            if len(self.current_fit) > 0:
                self.current_fit = self.current_fit[:len(self.current_fit)-1]
            if len(self.current_fit) > 0:
                self.best_fit = np.average(self.current_fit, axis=0)
