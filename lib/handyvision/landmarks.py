""" landmarks.py

Hand landmark helpers.
"""
import math
import numpy as np

from typing import Tuple, List
from enum import Enum, IntEnum, auto, unique

from google.protobuf.json_format import MessageToDict


@unique
class Handedness(str, Enum):
    """ 
    String enum representing handedness.
    Python 3.11 supports StrEnum, not the case with Python 3.10. 
    """
    LEFT = "LEFT"
    RIGHT = "RIGHT"


@unique
class HandLandmark(IntEnum):
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

@unique
class Finger(IntEnum):
    """
    Enum representing the fingers in a hand pose tuple.
    """
    THUMB = 0
    INDEX = auto()
    MIDDLE = auto()
    RING = auto()
    PINKY = auto()


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

    def get_point(self, index: int) -> np.ndarray:
        """ Get landmark from index as [x, y]
        """
        lmark = self.get(index)
        point = np.array([lmark.x, lmark.y])
        return point

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

class HandPose:
    """ Hand pose determined from HandState.
    """
    def __init__(self, digits: Tuple[bool] = None):
        """ 
        digits: tuple of bools where true = up, false = not up
        (thumb, index, middle, ring, pinky)
        """
        self.digits: List[bool] = [False] * 5
        if digits is not None and len(digits) == 5:
            self.digits = digits
    
    def set_digits(self, digits: List[bool]):
        """ Set state of all digits
        """
        if len(digits) != 5:
            return
        self.digits = digits

    def get_tuple(self) -> Tuple[bool, bool, bool, bool, bool]:
        """ Get the hand pose as a tuple of booleans representing 5 finger states
        """
        return tuple(self.digits)

    def get(self, finger: Finger):
        """ Get the state of a given finger
        """
        return self.digits[finger]
    
    def set(self, finger: Finger, state: bool):
        """ Set the state of a given finger
        """
        self.digits[finger] = state


class DetectDigitOptions:
    def __init__(self):
        self.thumb_extended_angle_threshold = 25
        self.wrist_finger_factor = 1.5 


def is_finger_extended(hand: HandState, finger: Finger, factor: int) -> bool:
    """ 
    Check if given finger is extended using given 
    wrist-tip-dist > (factor) * wrist-knuckle-dist.
    """
    match finger:
        case Finger.INDEX:
            return is_index_extended(hand, factor)
        case Finger.MIDDLE:
            return is_middle_extended(hand, factor)
        case Finger.RING:
            return is_ring_extended(hand, factor)
        case Finger.PINKY:
            return is_pinky_extended(hand, factor)
        case other:
            return None
    

def is_index_extended(hand: HandState, factor: int) -> bool:
    """ Check if index finger is extended
    """
    wrist = hand.get_point(HandLandmark.WRIST)
    index_knuckle = hand.get_point(HandLandmark.INDEX_FINGER_MCP)
    index_tip = hand.get_point(HandLandmark.INDEX_FINGER_TIP)

    wrist_index_k_dist = np.linalg.norm(wrist - index_knuckle)
    wrist_index_t_dist = np.linalg.norm(wrist - index_tip)

    if wrist_index_t_dist > factor * wrist_index_k_dist:
        return True
    
    return False


def is_middle_extended(hand: HandState, factor: int) -> bool:
    """ Check if middle finger is extended
    """
    wrist = hand.get_point(HandLandmark.WRIST)
    middle_knuckle = hand.get_point(HandLandmark.MIDDLE_FINGER_MCP)
    middle_tip = hand.get_point(HandLandmark.MIDDLE_FINGER_TIP)

    wrist_middle_k_dist = np.linalg.norm(wrist - middle_knuckle)
    wrist_middle_t_dist = np.linalg.norm(wrist - middle_tip)

    if wrist_middle_t_dist > factor * wrist_middle_k_dist:
        return True

    return False


def is_ring_extended(hand: HandState, factor: int) -> bool:
    """ Check if ring finger is extended
    """
    wrist = hand.get_point(HandLandmark.WRIST)
    ring_knuckle = hand.get_point(HandLandmark.RING_FINGER_MCP)
    ring_tip = hand.get_point(HandLandmark.RING_FINGER_TIP)

    wrist_ring_k_dist = np.linalg.norm(wrist - ring_knuckle)
    wrist_ring_t_dist = np.linalg.norm(wrist - ring_tip)

    if wrist_ring_t_dist > factor * wrist_ring_k_dist:
        return True
    
    return False


def is_pinky_extended(hand: HandState, factor: int) -> bool:
    """ Check if pinky finger is extended
    """
    wrist = hand.get_point(HandLandmark.WRIST)
    pinky_knuckle = hand.get_point(HandLandmark.PINKY_FINGER_MCP)
    pinky_tip = hand.get_point(HandLandmark.PINKY_FINGER_TIP)

    wrist_pinky_k_dist = np.linalg.norm(wrist - pinky_knuckle)
    wrist_pinky_t_dist = np.linalg.norm(wrist - pinky_tip)

    if wrist_pinky_t_dist > factor * wrist_pinky_k_dist:
        return True
    
    return False


def is_thumb_extended(hand: HandState, angle_threshold: int) -> bool:
    """ Check if thumb is extended
    """
    thumb_base = hand.get_point(HandLandmark.THUMB_CMC)
    thumb_tip = hand.get_point(HandLandmark.THUMB_TIP)
    index_k = hand.get_point(HandLandmark.INDEX_FINGER_MCP)

    thumb_tip_to_base = thumb_tip - thumb_base
    index_tip_to_base = index_k - thumb_base

    # Screams in **not** dot-product...
    thumb_angle_rad = \
        np.arctan2(thumb_tip_to_base[1], thumb_tip_to_base[0]) \
        - np.arctan2(index_tip_to_base[1], index_tip_to_base[0])
        
    thumb_angle_deg = np.abs(math.degrees(thumb_angle_rad))

    if thumb_angle_deg > 180.0:
        thumb_angle_deg = 360 - thumb_angle_deg

    if thumb_angle_deg > angle_threshold:
        return True
    
    return False

def non_thumb_fingers():
    """ Get non-thumb fingers
    """
    return [f for f in Finger if f != Finger.THUMB]
