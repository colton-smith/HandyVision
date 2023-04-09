import cv2 as cv
import mediapipe as mp
import handyvision as hv

from typing import Dict

def get_hand_states(hand_model, frame: cv.Mat):
    """ Pass frame through mediapipe and play with results 

    Return { LEFT: HandState, RIGHT: HandState } dict, with HandState == None 
    if it was not in the frame 
    """
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame_hands = hand_model.process(frame)

    hands: Dict[str, hv.HandState] = {
        hv.Handedness.LEFT.name: None, 
        hv.Handedness.RIGHT.name: None
    }

    if frame is None or frame.size == 0:
        return hands
    
    if frame_hands is None:
        return hands 

    # Question: does multi_hand_landmarks == None -> multi_handedness == None    
    if frame_hands.multi_hand_landmarks is None:
        return hands

    results = list(zip(frame_hands.multi_hand_landmarks, frame_hands.multi_handedness))
    for mp_landmark, mp_handedness in results:
        hand = hv.HandState(mp_landmark.landmark, mp_handedness)
        hands[hand.handedness.name] = hand
    
    return hands


def detect_digits(
        hand: hv.HandState, 
        opt: hv.DetectDigitOptions = hv.DetectDigitOptions()
    ) -> hv.HandPose:
    """ Determine what fingers are up given hand landmarks

    Optionally provide custom parameters.
    """
    hand_pose: hv.HandPose = hv.HandPose()
    thumb_extended = hv.is_thumb_extended(hand, opt.thumb_extended_angle_threshold)
    hand_pose.set(hv.Finger.THUMB, thumb_extended)

    for finger in hv.non_thumb_fingers():
        extended = hv.is_finger_extended(hand, finger, opt.wrist_finger_factor)
        hand_pose.set(finger, extended)

    return hand_pose

def main():
    # Open video capture using webcam (default)
    cam = hv.Camera(0)
    cam.set_auto_focus(False)
  
    # Set up hands model for image capture
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        model_complexity = 0,
        max_num_hands = 2,
        min_detection_confidence = 0.5,
        min_tracking_confidence = 0.5
    )

    while True:
        got_frame, frame = cam.get_frame()

        if not got_frame:
            continue

        frame = cv.flip(frame, 1)
        hand_data: Dict[str, hv.HandState] = get_hand_states(hands, frame)

        for name, data in hand_data.items():
            if data is None:
                continue
            else:
                print(f"{name} in frame!")
                pose: hv.HandPose = detect_digits(data)
                print(f"{name} digit states: {pose.get_tuple()}")
    
        # Display frame 
        cv.imshow("Hands", frame)
        if cv.waitKey(1) & 0xff == ord('q'):
            break

if __name__=="__main__":
    main()
