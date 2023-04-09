""" hand_pose_estimation.py

Hand pose estimation engine (hpee) client interface.

API: 
hpe = HPE(options)
hpe.update(frame)
g_left, g_right = hpe.gestures()
"""

import mediapipe as mp
import cv2 as cv

from typing import Tuple

from .gesture import *


def detect_digits(
        hand: HandState, 
        opt: DetectDigitOptions = DetectDigitOptions()
    ) -> HandPose:
    """ Determine what fingers are up given hand landmarks

    Optionally provide custom parameters.
    Return hand pose in NONE state if hand is None
    """
    hand_pose: HandPose = HandPose()
    if hand is None: 
        return hand_pose

    thumb_extended = is_thumb_extended(hand, opt.thumb_extended_angle_threshold)
    hand_pose.set(Finger.THUMB, thumb_extended)

    for finger in non_thumb_fingers():
        extended = is_finger_extended(hand, finger, opt.wrist_finger_factor)
        hand_pose.set(finger, extended)

    return hand_pose


class HPEEConfig:
    """ Hand pose estimation engine parameters
    """
    def __init__(
            self, 
            model_complexity: int = 0,
            max_hands: int = 2,
            min_detection_confidence: int = 0,
            min_tracking_confidence: int = 0,
            detect_digit_params: DetectDigitOptions = DetectDigitOptions()
        ):
        self.max_hands = max_hands
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.detect_digit_params = detect_digit_params


class HPEE:
    """ Hand pose estimation engine
    """
    def __init__(self, config: HPEEConfig = HPEEConfig()):
        self.__set_config(config)

        # Configure model
        self.model = mp.solutions.hands.Hands(
            self.model_complexity,
            self.max_hands,
            self.min_detection_confidence,
            self.min_tracking_confidence
        )

        # Left and right hand state
        self.left_hand_gesture: Gesture = None
        self.left_hand_pose: HandPose = None
        self.left_hand_state: HandState= None

        self.right_hand_gesture: Gesture = None
        self.right_hand_pose: HandPose = None
        self.right_hand_state: HandState = None

    def get_gesture_estimations(self) -> Tuple[Gesture, Gesture]:
        """ Return (left gesture, right gesture)
        """
        return self.left_hand_gesture, self.right_hand_gesture
    
    def get_gesture_estimation_strings(self) -> Tuple[Gesture, Gesture]:
        """ Return (left gesture, right gesture)
        """
        left = "UNKNOWN" if self.left_hand_gesture is None else self.left_hand_gesture.name
        right = "UNKNOWN" if self.right_hand_gesture is None else self.right_hand_gesture.name
        return left, right

    def update(self, frame: cv.Mat):
        """ Process RGB frame and update hand gesture estimations
        """
        self.__update_hand_states(frame)
        self.__update_pose_estimation()
        self.__update_hand_gesture()

    def __set_config(self, config: HPEEConfig):
        """ 
        Private.
        Apply configuration
        """
        self.max_hands = config.max_hands
        self.model_complexity = config.model_complexity
        self.min_detection_confidence = config.min_detection_confidence
        self.min_tracking_confidence = config.min_tracking_confidence
        self.detect_digit_params = config.detect_digit_params

    def __update_hand_states(self, frame: cv.Mat):
        """ 
        Private.
        Pass frame through mediapipe and update current hand state
        """
        frame_hands = self.model.process(frame)

        # Clear previous state
        self.left_hand_state = None
        self.right_hand_state = None 

        if frame is None or frame.size == 0:
            return
        
        if frame_hands is None:
            return

        # Question: does multi_hand_landmarks == None -> multi_handedness == None    
        if frame_hands.multi_hand_landmarks is None:
            return

        for mp_landmark, mp_handedness in zip(
                frame_hands.multi_hand_landmarks, 
                frame_hands.multi_handedness
            ):
            hand = HandState(mp_landmark.landmark, mp_handedness)
            if hand.handedness == Handedness.LEFT:
                self.left_hand_state = hand
            else:
                self.right_hand_state = hand
    
    def __update_pose_estimation(self):
        """ 
        Private.
        Process hand states and generate pose estimations
        """
        self.left_hand_pose = detect_digits(
            self.left_hand_state, 
            self.detect_digit_params
        )
        self.right_hand_pose = detect_digits(
            self.right_hand_state, 
            self.detect_digit_params
        )

    def __update_hand_gesture(self):
        """ 
        Private.
        Update hand gestures
        """
        self.left_hand_gesture = get_gesture(self.left_hand_pose)
        self.right_hand_gesture = get_gesture(self.right_hand_pose)
