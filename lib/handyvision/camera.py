""" camera.py

Opencv camera wrapper.
"""
import cv2 as cv
import numpy as np
from typing import Tuple

class Camera:
    """ Thin wrapper around cv.VideoCapture
    """
    def __init__(self, camera_idx: int = 0, name: str = None):
        self.index = camera_idx
        self.name = f"CAM_{self.index}" if name is None else name
        self.camera: cv.VideoCapture  = cv.VideoCapture(self.index)

    def good(self) -> bool:
        """ Check if the camera was initialized succesfully 
        """
        if self.camera is None:
                return False
        if not self.camera.isOpened():
             return False
        return True
        
    def get_frame(self) -> Tuple[bool, np.ndarray]:
        """ Capture frame from camera object 
        """
        return self.camera.read()
    
    def set_auto_focus(self, on: bool):
        """ Set auto focus
        """
        af_value = 1 if on else 0
        self.camera.set(cv.CAP_PROP_AUTOFOCUS, af_value)
    
    def set_resolution(self, width, height):
        """ Set resolution
        """
        self.camera.set(3, width)
        self.camera.set(4, height)

    def release(self):
        """ Release camera, also called when object is destroyed
        """
        self.camera.release()
