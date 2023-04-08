import cv2 as cv
import mediapipe as mp
import numpy as np
from google.protobuf.json_format import MessageToDict

def get_position(hands, frame, frame_height, frame_width, mp_hands):
    # Need to flip and convert frame for mp_hands to process the image
    frame = cv.flip(frame, 1)
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    
    frame_hands = hands.process(frame)

    # Convert back to original colour space for display
    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

    if frame_hands.multi_hand_landmarks: 

        label = ['','']
        label_num = 0
        for i in frame_hands.multi_handedness:
            label[label_num] = MessageToDict(i)['classification'][0]['label']
            label_num += 1
            if label_num > 1:
                break

        if len(frame_hands.multi_handedness) == 2:
        
            # 1, 4, 5 used for thumb -- 0, 5, 8 used for index -- 0, 9, 12 used for middle -- 0, 13, 16 used for ring -- 0, 17, 20 used for pinky
            return [frame_hands.multi_hand_landmarks[0].landmark[0].x * frame_width,    # 0
                    frame_hands.multi_hand_landmarks[0].landmark[0].y * frame_height,   # 1

                    frame_hands.multi_hand_landmarks[0].landmark[1].x * frame_width,    # 2
                    frame_hands.multi_hand_landmarks[0].landmark[1].y * frame_height,   # 3
                    frame_hands.multi_hand_landmarks[0].landmark[4].x * frame_width,    # 4
                    frame_hands.multi_hand_landmarks[0].landmark[4].y * frame_height,   # 5
 
                    frame_hands.multi_hand_landmarks[0].landmark[5].x * frame_width,    # 6
                    frame_hands.multi_hand_landmarks[0].landmark[5].y * frame_height,   # 7
                    frame_hands.multi_hand_landmarks[0].landmark[8].x * frame_width,    # 8
                    frame_hands.multi_hand_landmarks[0].landmark[8].y * frame_height,   # 9

                    frame_hands.multi_hand_landmarks[0].landmark[9].x * frame_width,    # 10
                    frame_hands.multi_hand_landmarks[0].landmark[9].y * frame_height,   # 11
                    frame_hands.multi_hand_landmarks[0].landmark[12].x * frame_width,   # 12
                    frame_hands.multi_hand_landmarks[0].landmark[12].y * frame_height,  # 13

                    frame_hands.multi_hand_landmarks[0].landmark[13].x * frame_width,   # 14
                    frame_hands.multi_hand_landmarks[0].landmark[13].y * frame_height,  # 15
                    frame_hands.multi_hand_landmarks[0].landmark[16].x * frame_width,   # 16
                    frame_hands.multi_hand_landmarks[0].landmark[16].y * frame_height,  # 17

                    frame_hands.multi_hand_landmarks[0].landmark[17].x * frame_width,   # 18
                    frame_hands.multi_hand_landmarks[0].landmark[17].y * frame_height,  # 19
                    frame_hands.multi_hand_landmarks[0].landmark[20].x * frame_width,   # 20
                    frame_hands.multi_hand_landmarks[0].landmark[20].y * frame_height,  # 21
                    label[0]], \
                   [frame_hands.multi_hand_landmarks[1].landmark[0].x * frame_width, 
                    frame_hands.multi_hand_landmarks[1].landmark[0].y * frame_height,

                    frame_hands.multi_hand_landmarks[1].landmark[1].x * frame_width, 
                    frame_hands.multi_hand_landmarks[1].landmark[1].y * frame_height,
                    frame_hands.multi_hand_landmarks[1].landmark[4].x * frame_width, 
                    frame_hands.multi_hand_landmarks[1].landmark[4].y * frame_height,

                    frame_hands.multi_hand_landmarks[1].landmark[5].x * frame_width, 
                    frame_hands.multi_hand_landmarks[1].landmark[5].y * frame_height,
                    frame_hands.multi_hand_landmarks[1].landmark[8].x * frame_width, 
                    frame_hands.multi_hand_landmarks[1].landmark[8].y * frame_height,

                    frame_hands.multi_hand_landmarks[1].landmark[9].x * frame_width, 
                    frame_hands.multi_hand_landmarks[1].landmark[9].y * frame_height,
                    frame_hands.multi_hand_landmarks[1].landmark[12].x * frame_width,
                    frame_hands.multi_hand_landmarks[1].landmark[12].y * frame_height,

                    frame_hands.multi_hand_landmarks[1].landmark[13].x * frame_width, 
                    frame_hands.multi_hand_landmarks[1].landmark[13].y * frame_height,
                    frame_hands.multi_hand_landmarks[1].landmark[16].x * frame_width, 
                    frame_hands.multi_hand_landmarks[1].landmark[16].y * frame_height,

                    frame_hands.multi_hand_landmarks[1].landmark[17].x * frame_width, 
                    frame_hands.multi_hand_landmarks[1].landmark[17].y * frame_height,
                    frame_hands.multi_hand_landmarks[1].landmark[20].x * frame_width, 
                    frame_hands.multi_hand_landmarks[1].landmark[20].y * frame_height,
                    label[1]]
        
        else:
            return [frame_hands.multi_hand_landmarks[0].landmark[0].x * frame_width, 
                    frame_hands.multi_hand_landmarks[0].landmark[0].y * frame_height,

                    frame_hands.multi_hand_landmarks[0].landmark[1].x * frame_width, 
                    frame_hands.multi_hand_landmarks[0].landmark[1].y * frame_height,
                    frame_hands.multi_hand_landmarks[0].landmark[4].x * frame_width, 
                    frame_hands.multi_hand_landmarks[0].landmark[4].y * frame_height,

                    frame_hands.multi_hand_landmarks[0].landmark[5].x * frame_width, 
                    frame_hands.multi_hand_landmarks[0].landmark[5].y * frame_height,
                    frame_hands.multi_hand_landmarks[0].landmark[8].x * frame_width, 
                    frame_hands.multi_hand_landmarks[0].landmark[8].y * frame_height,

                    frame_hands.multi_hand_landmarks[0].landmark[9].x * frame_width, 
                    frame_hands.multi_hand_landmarks[0].landmark[9].y * frame_height,
                    frame_hands.multi_hand_landmarks[0].landmark[12].x * frame_width,
                    frame_hands.multi_hand_landmarks[0].landmark[12].y * frame_height,

                    frame_hands.multi_hand_landmarks[0].landmark[13].x * frame_width, 
                    frame_hands.multi_hand_landmarks[0].landmark[13].y * frame_height,
                    frame_hands.multi_hand_landmarks[0].landmark[16].x * frame_width, 
                    frame_hands.multi_hand_landmarks[0].landmark[16].y * frame_height,

                    frame_hands.multi_hand_landmarks[0].landmark[17].x * frame_width, 
                    frame_hands.multi_hand_landmarks[0].landmark[17].y * frame_height,
                    frame_hands.multi_hand_landmarks[0].landmark[20].x * frame_width, 
                    frame_hands.multi_hand_landmarks[0].landmark[20].y * frame_height,
                    label[0]], np.zeros(23)
               
    return np.zeros(23),np.zeros(23)

def fingers_up(hand1, hand2):
    right = False
    left  = False
    if hand1[22] == 'Left':
        left_hand = hand1
        left = True
    elif hand1[22] == 'Right':
        right_hand = hand1
        right = True
    
    if hand2[22] == 'Left':
        left_hand = hand2
        left = True
    elif hand2[22] == 'Right':
        right_hand = hand2
        right = True

    # thumb, index, middle, ring, pinky
    left_fingers_up  = [None, None, None, None, None]
    right_fingers_up = [None, None, None, None, None]

    if left:
        # Thumb -->
        l_thumb_base = [left_hand[2], left_hand[3]]
        l_thumb_tip  = [left_hand[4], left_hand[5]]
        l_index_k    = [left_hand[6], left_hand[7]]
        l_thumb_angle_rad = np.arctan2(l_thumb_tip[1] - l_thumb_base[1], l_thumb_tip[0]-l_thumb_base[0]) - \
                            np.arctan2(l_index_k[1]-l_thumb_base[1], l_index_k[0]-l_thumb_base[0])
        l_thumb_angle_deg  = np.abs(l_thumb_angle_rad*180.0/np.pi)
        if l_thumb_angle_deg > 180.0:
            l_thumb_angle_deg = 360 - l_thumb_angle_deg

        # Index Finger --> 0, 6, 1, 7 = knuckle -- 0, 8, 1, 9 = tip
        l_wrist_index_k = np.sqrt(((left_hand[0] - left_hand[6])**2) + ((left_hand[1] - left_hand[7])**2))
        l_wrist_index_t = np.sqrt(((left_hand[0] - left_hand[8])**2) + ((left_hand[1] - left_hand[9])**2))

        # Middle Finger --> 0, 10, 1, 11 = knuckle -- 0, 12, 1, 13 = tip
        l_wrist_middle_k = np.sqrt(((left_hand[0] - left_hand[10])**2) + ((left_hand[1] - left_hand[11])**2))
        l_wrist_middle_t = np.sqrt(((left_hand[0] - left_hand[12])**2) + ((left_hand[1] - left_hand[13])**2))

        # Ring Finger --> 0, 14, 1, 15 = knuckle -- 0, 16, 1, 17 = tip
        l_wrist_ring_k = np.sqrt(((left_hand[0] - left_hand[14])**2) + ((left_hand[1] - left_hand[15])**2))
        l_wrist_ring_t = np.sqrt(((left_hand[0] - left_hand[16])**2) + ((left_hand[1] - left_hand[17])**2))

        # Pinky Finger --> 0, 18, 1, 19 = knuckle -- 0, 20, 1, 21 = tip
        l_wrist_pinky_k = np.sqrt(((left_hand[0] - left_hand[18])**2) + ((left_hand[1] - left_hand[19])**2))
        l_wrist_pinky_t = np.sqrt(((left_hand[0] - left_hand[20])**2) + ((left_hand[1] - left_hand[21])**2))

        if l_thumb_angle_deg > 40:
            left_fingers_up[0] = True
        else : left_fingers_up[0] = False

        if l_wrist_index_t > 1.5 * l_wrist_index_k:
            left_fingers_up[1] = True
        else : left_fingers_up[1] = False

        if l_wrist_middle_t > 1.5 * l_wrist_middle_k:
            left_fingers_up[2] = True
        else : left_fingers_up[2] = False

        if l_wrist_ring_t > 1.5 * l_wrist_ring_k:
            left_fingers_up[3] = True
        else : left_fingers_up[3] = False

        if l_wrist_pinky_t > 1.5 * l_wrist_pinky_k:
            left_fingers_up[4] = True
        else : left_fingers_up[4] = False
    
    if right:
        # Thumb -->
        r_thumb_base = [right_hand[2], right_hand[3]]
        r_thumb_tip  = [right_hand[4], right_hand[5]]
        r_index_k    = [right_hand[6], right_hand[7]]
        r_thumb_angle_rad = np.arctan2(r_thumb_tip[1] - r_thumb_base[1], r_thumb_tip[0]-r_thumb_base[0]) - \
                            np.arctan2(r_index_k[1]-r_thumb_base[1], r_index_k[0]-r_thumb_base[0])
        r_thumb_angle_deg  = np.abs(r_thumb_angle_rad*180.0/np.pi)
        if r_thumb_angle_deg > 180.0:
            r_thumb_angle_deg = 360 - r_thumb_angle_deg

        # Index Finger --> 0, 6, 1, 7 = knuckle -- 0, 8, 1, 9 = tip
        r_wrist_index_k = np.sqrt(((right_hand[0] - right_hand[6])**2) + ((right_hand[1] - right_hand[7])**2))
        r_wrist_index_t = np.sqrt(((right_hand[0] - right_hand[8])**2) + ((right_hand[1] - right_hand[9])**2))

        # Middle Finger --> 0, 10, 1, 11 = knuckle -- 0, 12, 1, 13 = tip
        r_wrist_middle_k = np.sqrt(((right_hand[0] - right_hand[10])**2) + ((right_hand[1] - right_hand[11])**2))
        r_wrist_middle_t = np.sqrt(((right_hand[0] - right_hand[12])**2) + ((right_hand[1] - right_hand[13])**2))

        # Ring Finger --> 0, 14, 1, 15 = knuckle -- 0, 16, 1, 17 = tip
        r_wrist_ring_k = np.sqrt(((right_hand[0] - right_hand[14])**2) + ((right_hand[1] - right_hand[15])**2))
        r_wrist_ring_t = np.sqrt(((right_hand[0] - right_hand[16])**2) + ((right_hand[1] - right_hand[17])**2))

        # Pinky Finger --> 0, 18, 1, 19 = knuckle -- 0, 20, 1, 21 = tip
        r_wrist_pinky_k = np.sqrt(((right_hand[0] - right_hand[18])**2) + ((right_hand[1] - right_hand[19])**2))
        r_wrist_pinky_t = np.sqrt(((right_hand[0] - right_hand[20])**2) + ((right_hand[1] - right_hand[21])**2))

        if r_thumb_angle_deg > 45:
            right_fingers_up[0] = True
        else : right_fingers_up[0] = False

        if r_wrist_index_t > 1.5 * r_wrist_index_k:
            right_fingers_up[1] = True
        else : right_fingers_up[1] = False

        if r_wrist_middle_t > 1.5 * r_wrist_middle_k:
            right_fingers_up[2] = True
        else : right_fingers_up[2] = False

        if r_wrist_ring_t > 1.5 * r_wrist_ring_k:
            right_fingers_up[3] = True
        else : right_fingers_up[3] = False

        if r_wrist_pinky_t > 1.5 * r_wrist_pinky_k:
            right_fingers_up[4] = True
        else : right_fingers_up[4] = False

    return left_fingers_up, right_fingers_up

def get_gesture(left_fingers, right_fingers):
    gesture_left = 'unrecognized gesture'
    gesture_right = 'unrecognized gesture'

    # Finding Left Gesture
    if   np.array_equiv(left_fingers, [False, False, False, False, False]) : gesture_left = 'fist'
    elif np.array_equiv(left_fingers, [False, True , False, False, False]) : gesture_left = 'point'
    elif np.array_equiv(left_fingers, [False, True , True , False, False]) : gesture_left = 'peace'
    elif np.array_equiv(left_fingers, [False, True , True , True , False]) : gesture_left = 'three'
    elif np.array_equiv(left_fingers, [False, True , True , True , True ]) : gesture_left = 'four'
    elif np.array_equiv(left_fingers, [True , True , True , True , True ]) : gesture_left = 'spread'
    elif np.array_equiv(left_fingers, [True , False, False, False, False]) : gesture_left = 'thumb'
    elif np.array_equiv(left_fingers, [False, True , False, False, True ]) : gesture_left = 'rock'
    elif np.array_equiv(left_fingers, [True , True , False, False, True ]) : gesture_left = 'love'
    elif np.array_equiv(left_fingers, [True , True , True , False, False]) : gesture_left = 'gun'
    elif np.array_equiv(left_fingers, [False, False, True , True , True ]) : gesture_left = 'splash'
    elif np.array_equiv(left_fingers, [False, False, False, False, True ]) : gesture_left = 'pinky'
    elif np.array_equiv(left_fingers, [True , False, False, False, True ]) : gesture_left = 'hang loose'
    elif np.array_equiv(left_fingers, [False, False, True , False, False]) : gesture_left = 'flipoff'
    elif np.array_equiv(left_fingers, [None , None , None , None , None ]) : gesture_left = 'not in frame'
    else : gesture_left = 'unrecognized gesture'

    # Finding right gesture
    if   np.array_equiv(right_fingers, [False, False, False, False, False]) : gesture_right = 'fist'
    elif np.array_equiv(right_fingers, [False, True , False, False, False]) : gesture_right = 'point'
    elif np.array_equiv(right_fingers, [False, True , True , False, False]) : gesture_right = 'peace'
    elif np.array_equiv(right_fingers, [False, True , True , True , False]) : gesture_right = 'three'
    elif np.array_equiv(right_fingers, [False, True , True , True , True ]) : gesture_right = 'four'
    elif np.array_equiv(right_fingers, [True , True , True , True , True ]) : gesture_right = 'spread'
    elif np.array_equiv(right_fingers, [True , False, False, False, False]) : gesture_right = 'thumb'
    elif np.array_equiv(right_fingers, [False, True , False, False, True ]) : gesture_right = 'rock'
    elif np.array_equiv(right_fingers, [True , True , False, False, True ]) : gesture_right = 'love'
    elif np.array_equiv(right_fingers, [True , True , True , False, False]) : gesture_right = 'gun'
    elif np.array_equiv(right_fingers, [False, False, True , True , True ]) : gesture_right = 'splash'
    elif np.array_equiv(right_fingers, [False, False, False, False, True ]) : gesture_right = 'pinky'
    elif np.array_equiv(right_fingers, [True , False, False, False, True ]) : gesture_right = 'hang loose'
    elif np.array_equiv(right_fingers, [False, False, True , False, False]) : gesture_right = 'flipoff'
    elif np.array_equiv(right_fingers, [None , None , None , None , None ]) : gesture_right = 'not in frame'
    else : gesture_right = 'unrecognized gesture'

    return gesture_left, gesture_right

def main():

    hands_p1 = mp.solutions.hands
    hands_p2 = mp.solutions.hands

    # Open video capture using webcam (default)
    capture = cv.VideoCapture(0)
    got_frame, frame = capture.read()
    frame_height, frame_width, c = frame.shape 

    mp_hands_p1 = mp.solutions.hands
    mp_hands_p2 = mp.solutions.hands

    # Setting up hands model for video capture
    hands_p1 = mp_hands_p1.Hands(
        model_complexity = 0,
        max_num_hands = 2,
        min_detection_confidence = 0.5,
        min_tracking_confidence = 0.5
    )

    hands_p2 = mp_hands_p1.Hands(
        model_complexity = 0,
        max_num_hands = 2,
        min_detection_confidence = 0.5,
        min_tracking_confidence = 0.5
    )

    while True:
        got_frame, frame = got_frame, frame = capture.read()
        
        # These are backwards from expected because we have to flip the frame so when it displays it looks like a mirror
        frame_p2 = frame[:, 0:frame_width//2]
        frame_p1 = frame[:, frame_width//2:frame_width-1]
        
        hand1_p1, hand2_p1 = get_position(hands_p1, frame_p1, frame_height, frame_width//2, mp_hands_p1)
        hand1_p2, hand2_p2 = get_position(hands_p2, frame_p2, frame_height, frame_width//2, mp_hands_p2)

        left_fingers_p1, right_fingers_p1 = fingers_up(hand1_p1, hand2_p1)
        left_fingers_p2, right_fingers_p2 = fingers_up(hand1_p2, hand2_p2)

        gesture_name_l_p1, gesture_name_r_p1 = get_gesture(left_fingers_p1, right_fingers_p1)
        gesture_name_l_p2, gesture_name_r_p2 = get_gesture(left_fingers_p2, right_fingers_p2)

        # Any test to be written on the frame should be written after this flip
        frame = cv.flip(frame, 1)
        coordinates_l_p1 = (50,50)
        coordinates_r_p1 = (50,100)
        coordinates_l_p2 = (50 + frame_width//2, 50)
        coordinates_r_p2 = (50 + frame_width//2, 100)
        font = cv.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (255,0,255)
        thickness = 2
        frame = cv.putText(frame, gesture_name_l_p1, coordinates_l_p1, font, fontScale, color, thickness, cv.LINE_AA)
        frame = cv.putText(frame, gesture_name_r_p1, coordinates_r_p1, font, fontScale, color, thickness, cv.LINE_AA)
        frame = cv.putText(frame, gesture_name_l_p2, coordinates_l_p2, font, fontScale, color, thickness, cv.LINE_AA)
        frame = cv.putText(frame, gesture_name_r_p2, coordinates_r_p2, font, fontScale, color, thickness, cv.LINE_AA)

        if gesture_name_l_p1 == 'flipoff' or gesture_name_r_p1 == 'flipoff':
            frame[:, frame_width//2:frame_width-1, 2] = 255

        if gesture_name_l_p2 == 'flipoff' or gesture_name_r_p2 == 'flipoff':
            frame[:, 0:frame_width//2, 2] = 255

        cv.imshow("Game Window", frame)
        
        if cv.waitKey(1) & 0xff == ord('q'):
            capture.release()

if __name__=="__main__":
    main()
    
