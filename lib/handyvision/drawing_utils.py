""" drawing_utils.py

Opencv drawing utilities. 
"""
import cv2 as cv
from enum import IntEnum, unique


@unique
class HorizontalHalf(IntEnum):
    LEFT = 0
    RIGHT = 1


class Font:
    def __init__(self, face, scale, thickness, color):
        self.face = face
        self.scale = scale
        self.thickness = thickness
        self.color = color


class LineOptions: 
    def __init__(self, color = None, thickness = None):
        default_color = (255, 255, 255)
        default_thickness = 4

        self.color = color if color is not None else default_color 
        self.thickness = thickness if thickness is not None else default_thickness 


class GaussianConfig:
    def __init__(self):
        self.ksize = (3,3)
        self.sigmaX = 2
        self.sigmaY = 2


def draw_center_line(frame: cv.Mat, options: LineOptions) -> cv.Mat:
    """ Draw center line on frame with options
    """
    h = frame.shape[0]
    w = frame.shape[1]
    frame = cv.line(frame, (w//2, 0), (w//2, h-1), options.color, options.thickness)
    return frame


def draw_text_centered(frame: cv.Mat, text: str, font: Font):
    """ Draw text centered in the frame, return frame, text_size
    """
    rows = frame.shape[0]
    text_size, baseline = cv.getTextSize(text, font.face, font.scale, font.thickness)
    
    fh, fw, _ = frame.shape
    origin = fw // 2, fh // 2

    text_left = origin[0] - text_size[0] // 2
    text_top = origin[1] + text_size[1] // 2 - baseline

    frame = cv.putText(
        frame, 
        text, 
        org=(text_left, text_top), 
        fontFace=font.face,
        fontScale=font.scale,
        thickness=font.thickness,
        color=font.color
    )

    return frame, text_size


def draw_text_bottom_left(frame: cv.Mat, text: str, font: Font):
    """ Draw text anchored at bottom left of frame, return frame, text_size
    """
    rows = frame.shape[0]
    text_size, baseline = cv.getTextSize(text, font.face, font.scale, font.thickness)
    frame = cv.putText(
        frame, 
        text, 
        org=(0, rows - 1 - baseline), 
        fontFace=font.face,
        fontScale=font.scale,
        thickness=font.thickness,
        color=font.color
    )

    return frame, text_size


def draw_text_top_right(frame: cv.Mat, text: str, font: Font):
    """ Draw text at the top right of the frame
    """
    text_size, baseline = cv.getTextSize(text, font.face, font.scale, font.thickness)
    fh, fw, _ = frame.shape

    text_left =  fw - text_size[0] - 1
    text_bot = text_size[1] + baseline 

    frame = cv.putText(
        frame, 
        text, 
        org=(text_left, text_bot), 
        fontFace=font.face,
        fontScale=font.scale,
        thickness=font.thickness,
        color=font.color
    )

    return frame, text_size
