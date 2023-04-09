""" img_utils.py

Misc. image processing utilities.
"""
import cv2 as cv
from typing import Tuple


def vertically_bisect_image(img: cv.Mat) -> Tuple[cv.Mat, cv.Mat]:
    """ Return the left and right halves of a 2D image
    """
    h = img.shape[0]
    w = img.shape[1]

    left_half = img[:, 0 : w//2]
    right_half = img[:, w//2:w-1]
    
    return left_half, right_half
