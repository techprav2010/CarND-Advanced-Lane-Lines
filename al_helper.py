import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import al_consts
import matplotlib.gridspec as gridspec
import re

# matplotlib wrapper for ploting and saving images
class AlHelper:

    ####################################################################
    def _get_file_name(self,lbl, title=""):
        name = title.rstrip() + "_" + lbl.rstrip()
        name = "".join([ c if (c.isalpha()  or c.isnumeric()) else "_" for c in name ])
        name = name.replace("__", "_",100)
        return "output_images/"+name.rstrip()+".png"

    def _addImg(self, img, lbl,row, n,col, cmap=None, title="", ):
        ax = plt.subplot(row, n, col)
        plt.imshow(img, cmap=cmap)
        plt.xlabel(lbl)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        file=self._get_file_name(lbl, title=title)
        #also save the image
        mpimg.imsave(file, img, cmap=cmap)

    def display(self, images, non_gray={}, size=7, cols=2, cmap=None, title="img"):
        # display helper -- given an hashtable it draws images
        total= len(images)
        has_remind = total % cols
        rows = total / cols
        if has_remind :
            rows += 1
        rows += 1
        fig = plt.figure(figsize=(size*cols,size*rows ) , edgecolor="blue", facecolor="lightgreen" )
        plt.grid(axis='both')
        counter=1
        for imag in images:
            self._addImg(images[imag], imag, rows, cols, counter, cmap='gray', title=title)
            counter =counter+1
        plt.suptitle(title)
        plt.tight_layout( 0.1,0.1, 0.1)

        plt.subplots_adjust(wspace=0.1, hspace=0)
        plt.show()

    def display2(self, img1, lbl1, img2, lbl2,  x=14, y= 7, cmap=None,title="img"):
        non_gray = {}
        images = {}
        images[lbl1] = img1
        images[lbl2] = img2
        self.display(images, non_gray, size=7, cols=2,cmap=cmap, title=title)


    ####################################################################


