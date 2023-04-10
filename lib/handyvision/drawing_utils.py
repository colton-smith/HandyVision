""" drawing_utils.py

Opencv drawing utilities. 
"""
import cv2 as cv

from enum import IntEnum, unique


COLORS = {
    "SLATE_GREY": (79, 83, 88),
    "WHITE": (255, 255, 255),
    "BLACK": (0, 0, 0)
}


@unique
class HorizontalHalf(IntEnum):
    LEFT = 0
    RIGHT = 1


class Rect:
    def __init__(self, width, height, color, thickness, line_type):
        self.width = width
        self.height = height
        self.color = color
        self.thickness = thickness 
        self.line_type = line_type


class Font:
    def __init__(self, face, scale, thickness, color):
        self.face = face
        self.scale = scale
        self.thickness = thickness
        self.color = color


class Line: 
    def __init__(self, color, thickness):
        self.color = color
        self.thickness = thickness


class GaussianConfig:
    def __init__(self):
        self.ksize = (3,3)
        self.sigmaX = 2
        self.sigmaY = 2


def draw_center_line(frame: cv.Mat, options: Line) -> cv.Mat:
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


def draw_text_top_left(frame: cv.Mat, text: str, font: Font):
    """ Draw text anchored at top left of frame, return frame, text_size
    """
    rows = frame.shape[0]
    text_size, baseline = cv.getTextSize(text, font.face, font.scale, font.thickness)
    frame = cv.putText(
        frame, 
        text, 
        org=(0, text_size[1] + baseline), 
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


def draw_text_bottom_right(frame: cv.Mat, text: str, font: Font):
    """ Draw text at the bottom right of the frame
    """
    text_size, baseline = cv.getTextSize(text, font.face, font.scale, font.thickness)
    fh, fw, _ = frame.shape

    text_left = fw - text_size[0] - 1
    text_bot = fh - baseline 

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


def get_text_bot_left(text_size, baseline, center):
    """ Return the bot left coord for text from the center point and size
    """
    return center[0] - text_size[0] // 2, center[1] + text_size[1] // 2 - baseline


def get_text_size(text: str, font: Font):
    """ Get prospective size of text box (and baseline)
    """
    return cv.getTextSize(text, font.face, font.scale, font.thickness)


def draw_text(frame: cv.Mat, text: str, font: Font, bottom_left):
    """ Draw text at point (having already adjusted for size and baseline)
    """
    frame = cv.putText(
        frame, 
        text, 
        org=bottom_left, 
        fontFace=font.face,
        fontScale=font.scale,
        thickness=font.thickness,
        color=font.color
    )

    return frame


def draw_rectangle(frame: cv.Mat, rect: Rect, top_left):
    """ Draw rectangle on frame at point
    """
    bottom_right = top_left[0] + rect.width, top_left[1] + rect.height
    return cv.rectangle(frame, top_left, bottom_right, rect.color, rect.thickness, rect.line_type)
