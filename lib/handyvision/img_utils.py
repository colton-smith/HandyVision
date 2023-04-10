""" img_utils.py

Misc. image processing utilities.
"""
import cv2 as cv
import numpy as np
from typing import Tuple


def overlay_transparent_image(background: cv.Mat, overlay: cv.Mat, top_left: np.array):
    """ Overlay an image with a transparency channel over an image without one

    Note: does not handle blending, just binary transparency. 
    """
    overlay_alpha = overlay[:,:,3] / 255.0
    image_alpha = 1 - overlay_alpha 

    overlay_size = np.array([overlay.shape[0], overlay.shape[1]])
    bot_right = top_left + overlay_size

    background[top_left[0]:bot_right[0], top_left[1]:bot_right[1], 0:3] = \
        np.dstack([image_alpha, image_alpha, image_alpha]) * background[top_left[0]:bot_right[0], top_left[1]:bot_right[1], 0:3] \
        + np.dstack([overlay_alpha, overlay_alpha, overlay_alpha]) * overlay[: , : , 0:3] \

    return background


def vertically_bisect_image(img: cv.Mat) -> Tuple[cv.Mat, cv.Mat]:
    """ Return the left and right halves of a 2D image
    """
    h = img.shape[0]
    w = img.shape[1]

    left_s = np.s_[:, 0 : w//2]
    right_s = np.s_[:, w//2:w-1]

    left_half = img[left_s]
    right_half = img[right_s]
    
    return left_half, right_half, left_s, right_s


def bgr2rgb(img: cv.Mat): 
    """ Convert image from gbr to rbg 
    """
    return cv.cvtColor(img, cv.COLOR_BGR2RGB)


def rgb2greyscale(img: cv.Mat):
    """ Convert image from rgb to greyscale 
    """
    return cv.cvtColor(img, cv.COLOR_RGB2GRAY)


def bgr2greyscale(img: cv.Mat):
    """ Convert image to greyscale 
    """
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

def flip(img: cv.Mat, flip_code: int):
    """ Wrapper around cv.flip
    """
    return cv.flip(img, flip_code)
