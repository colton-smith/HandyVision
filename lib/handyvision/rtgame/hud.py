""" hud.py

Heads up display drawing utilities. 
"""
import cv2 as cv
import numpy as np
import handyvision as hv
import handyvision.rtgame as rtg


class HUD:
    """ Track game HUD elements
    """
    def __init__(self):
        pass

    def draw_fps(frame: cv.Mat) -> cv.Mat:
        """ Draw FPS
        """
        pass


def draw_gesture_icons(frame: cv.Mat, left: cv.Mat, right: cv.Mat) -> cv.Mat:
    """ Draw left and right gesture icons on the frame
    """
    center_col = frame.shape[1] // 2
    overlay_w = left.shape[1]
    inter_icon_spacer_half_width = 5

    # Vertical bumper from top of screen
    vertical_padding = 10

    # Offset from center width of overlay + half the spacer width
    start_col = center_col - overlay_w - inter_icon_spacer_half_width
    start_row = vertical_padding

    left_image_top_left = np.array([start_row, start_col])
    frame = hv.overlay_transparent_image(frame, left, left_image_top_left)

    # Offset right image width of the left overlay, + 2 * spacer
    right_image_top_left = left_image_top_left + (0, overlay_w + 2 * inter_icon_spacer_half_width)
    frame = hv.overlay_transparent_image(frame, right, right_image_top_left)

    return frame


def muddle_frame(frame: cv.Mat, half: hv.HorizontalHalf, config: rtg.GameConfig):
    """ Muddle one half of the frame 

    Applies a gaussian filter and tints the screen frame red
    """
    _, cols, _ = frame.shape

    # Tint + Blur
    blur_options =config.flip_off_blur_config
    ksize = blur_options.ksize
    sx = blur_options.sigmaX
    sy = blur_options.sigmaY

    if half == hv.HorizontalHalf.LEFT:
        frame[:, 0:cols//2, 2] = 255
        frame[:, 0:cols//2, :] = cv.GaussianBlur(frame[:, 0:cols//2, :], ksize, sx, sy)
    else:
        frame[:, cols//2:cols, 2] = 255
        frame[:, cols//2:cols, :] = cv.GaussianBlur(frame[:, cols//2:cols, :], ksize, sx, sy)

    return frame
