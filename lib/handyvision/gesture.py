""" gesture.py

Hand gesture utilities.
"""
import os

import cv2 as cv
import numpy as np

import json

from typing import Dict
from enum import Enum, unique
from .landmarks import *


@unique
class Gesture(str, Enum):
    """ String enum representing gestures.
    """
    ARTHRITIS = "ARTHRITIS"
    FIST = "FIST"
    POINT = "POINT"
    PEACE = "PEACE"
    THREE = "THREE"
    FOUR = "FOUR"
    SPREAD = "SPREAD"
    THUMB = "THUMB"
    ROCK = "ROCK"
    LOVE = "LOVE"
    GUN_DOUBLE = "GUN_DOUBLE"
    GUN_SINGLE = "GUN_SINGLE"
    SPLASH = "SPLASH"
    CHEERIO = "CHEERIO"
    HANG_LOOSE = "HANG_LOOSE"
    FLIPOFF =  "FLIPOFF"
    OUT_OF_FRAME = "OUT_OF_FRAME"


gesture_map = {
    (False, True, False, True, False): Gesture.ARTHRITIS,
    (False, False, False, False, False): Gesture.FIST,
    (False, True , False, False, False): Gesture.POINT,
    (False, True , True , False, False): Gesture.PEACE,
    (False, True , True , True , False): Gesture.THREE,
    (False, True , True , True , True): Gesture.FOUR,
    (True , True , True , True , True): Gesture.SPREAD,
    (True , False, False, False, False): Gesture.THUMB,
    (False, True , False, False, True): Gesture.ROCK,
    (True , True , False, False, True): Gesture.LOVE,
    (True , True , True , False, False): Gesture.GUN_DOUBLE,
    (True , True , False , False, False): Gesture.GUN_SINGLE,
    (False, False, True , True , True): Gesture.SPLASH,
    (False, False, False, False, True): Gesture.CHEERIO,
    (True , False, False, False, True): Gesture.HANG_LOOSE,
    (False, False, True , False, False): Gesture.FLIPOFF,
    (None , None , None , None , None): Gesture.OUT_OF_FRAME,
}

class IconManager:
    """ Icon manager 

    Pass a relative or absolute path to a folder containing a manifest.json file.
    """
    def __init__(self, asset_folder_path: str):
        self.asset_folder = os.path.normpath(asset_folder_path)
        self.asset_manifest = os.path.join(self.asset_folder, "manifest.json")
        self.gesture_icon_map_left = {}
        self.gesture_icon_map_right = {}
        self.__load_assets()

    def icon_for_gesture(self, handedness: Handedness, gesture: Gesture) -> cv.Mat:
        """ Get image for handedness and gesture

        Returns None if there is no icon for gesture.
        """
        if handedness == Handedness.LEFT and gesture in self.gesture_icon_map_left:
            return self.gesture_icon_map_left[gesture]
        elif handedness == Handedness.RIGHT and gesture in self.gesture_icon_map_right:
            return self.gesture_icon_map_right[gesture]
        else:
            return None

    def __load_assets(self):
        """ Load assets from the provided manifest file 
        """
        with open(self.asset_manifest) as f:
            data = json.load(f)
            gesture_data = data["gestures"]

            left_folder = gesture_data["LEFT_FOLDER"]
            right_folder = gesture_data["RIGHT_FOLDER"]

            for mapping in gesture_data["PATH_GESTURE_MAPPING"]:
                file = mapping["File"]
                gesture = mapping["Gesture"]
                left_file = os.path.join(self.asset_folder, left_folder, file)
                right_file = os.path.join(self.asset_folder, right_folder, file)
                left_image = cv.imread(left_file)
                right_image = cv.imread(right_file)
                self.gesture_icon_map_left[Gesture[gesture]] = left_image
                self.gesture_icon_map_right[Gesture[gesture]] = right_image


def get_gesture(hand_pose: HandPose) -> Gesture:
    """ Get gesture associated with hand pose

    Return None if gesture is unrecognized.
    """
    global gesture_map
    finger_states = hand_pose.get_tuple()

    if not finger_states in gesture_map:
        return None
    
    return gesture_map[hand_pose.get_tuple()]


def get_gesture_string(hand_pose: HandPose) -> str:
    """ Get gesture string associated with hand pose

    Return "UNKOWN" if gesture is unrecognized.
    """
    gesture = get_gesture(hand_pose)
    if gesture is None:
        return "UNKNOWN"
    else:
        return gesture.name


def get_string_from_gesture(gesture: Gesture) -> str:
    """ Get string from gesture
    """
    if gesture is None:
        return "NONE"
    else:
        return gesture.name

def get_random_gesture() -> Gesture:
    """ Return a random gesture 
    """
    possible_gestures = [g for g in Gesture]
    return possible_gestures[np.random.randint(0, len(possible_gestures))]
