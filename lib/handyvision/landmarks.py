""" landmarks.py

Hand landmark helpers.
"""
from enum import Enum, IntEnum, auto, unique
from google.protobuf.json_format import MessageToDict

import numpy as np


@unique
class Handedness(str, Enum):
    """ 
    String enum representing handedness.
    Python 3.11 supports StrEnum, not the case with Python 3.10. 
    """
    LEFT = "LEFT"
    RIGHT = "RIGHT"


@unique
class HandLandmarkName(IntEnum):
    """ 
    Mediapipe hand landmark names according to: 
    https://developers.google.com/mediapipe/solutions/vision/hand_landmarker

    From base of thumb to tip: 
        CMC -> MCP -> IP -> TIP
    
    From base of finger to tip:
        MCP -> PIP -> DIP -> TIP
    """
    WRIST = 0

    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4

    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8

    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12

    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16

    PINKY_FINGER_MCP = 17
    PINKY_FINGER_PIP = 18
    PINKY_FINGER_DIP = 19
    PINKY_FINGER_TIP = 20


def get_handedness(mp_handedness) -> Handedness:
    """ Extract handedness from media pipe handedness output 
    """
    label: str = MessageToDict(mp_handedness)["classification"][0]["label"]
    label = label.upper()
    return Handedness[label]


class HandState:
    """ 
    Struct representing all hand landmarks generated 
    by mediapipe.

    landmarks: output.landmark 
    handedness: output.handedness
    """
    def __init__(self, landmarks, handedness):
        self.handedness = get_handedness(handedness)
        self.landmarks = landmarks

    def get(self, index: int):
        """ Get landmark from index
        """
        return self.landmarks[index]

    def is_left(self):
        """ Return true if these landmarks are for a left hand
        """
        if self.handedness == Handedness.LEFT:
            return True
        return False
    
    def is_right(self):
        """ Return true if these landmarks are for a right hand
        """
        if self.handedness == Handedness.RIGHT:
            return True
        return False
