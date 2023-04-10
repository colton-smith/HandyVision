""" utils.py

Utility classes for reaction time game (RTG)
"""
import handyvision as hv
import cv2 as cv

from enum import unique, IntEnum, auto


@unique 
class GameState(IntEnum):
    """ Game state enum
    """
    IDLE = 0
    COUNTDOWN = auto()
    DISPLAY_GESTURE = auto()
    CHECK_GESTURE = auto()
    WIN_SCREEN = auto()


class GameConfig:
    def __init__(self):
        self.p1_label_color = (255, 255, 255)
        self.p2_label_color = (255, 255, 255)
        self.gesture_icon_scale = (0.5, 0.5)
        self.skeleton_toggle_key = 's'
        self.fps_count_toggle_key= 'f'
        self.draw_center_line = False
        self.flip_off_blur_config = hv.GaussianConfig()
        self.font = cv.FONT_HERSHEY_SIMPLEX
        self.start_game_count = 5
        self.default_num_rounds = 4
        self.gest_to_rounds = {
            hv.Gesture.POINT : 2,
            hv.Gesture.PEACE : 4,
            hv.Gesture.THREE : 6,
            hv.Gesture.FOUR : 8,
            hv.Gesture.SPREAD : 10
        }
