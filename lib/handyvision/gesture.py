""" gesture.py

Hand gesture utilities.
"""

from enum import Enum, unique
from .landmarks import *


@unique
class Gesture(str, Enum):
    """ String enum representing gestures.
    """
    FIST = "FIST"
    POINT = "POINT"
    PEACE = "PEACE"
    THREE = "THREE"
    FOUR = "FOUR"
    SPREAD = "SPREAD"
    THUMB = "THUMB"
    ROCK = "ROCK"
    LOVE = "LOVE"
    GUN = "GUN"
    SPLASH = "SPLASH"
    CHEERIO = "CHEERIO"
    HANG_LOOSE = "HANG_LOOSE"
    FLIPOFF =  "FLIPOFF"
    OUT_OF_FRAME = "OUT_OF_FRAME"


gesture_map = {
    (False, False, False, False, False): Gesture.FIST,
    (False, True , False, False, False): Gesture.POINT,
    (False, True , True , False, False): Gesture.PEACE,
    (False, True , True , True , False): Gesture.THREE,
    (False, True , True , True , True): Gesture.FOUR,
    (True , True , True , True , True): Gesture.SPREAD,
    (True , False, False, False, False): Gesture.THUMB,
    (False, True , False, False, True): Gesture.ROCK,
    (True , True , False, False, True): Gesture.LOVE,
    (True , True , True , False, False): Gesture.GUN,
    (False, False, True , True , True): Gesture.SPLASH,
    (False, False, False, False, True): Gesture.CHEERIO,
    (True , False, False, False, True): Gesture.HANG_LOOSE,
    (False, False, True , False, False): Gesture.FLIPOFF,
    (None , None , None , None , None): Gesture.OUT_OF_FRAME,
}


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