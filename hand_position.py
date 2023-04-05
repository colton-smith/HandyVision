import cv2 as cv
import mediapipe as mp
import numpy as np
from google.protobuf.json_format import MessageToDict

def get_position(hands, capture, frame_height, frame_width, mp_hands):
    got_frame, frame = capture.read() # Get frame to analyze

    # Making sure capture got valid frame before trying to analyze
    if not got_frame:
        print("Empty Frame")
        

    # Need to flip and convert frame for mp_hands to process the image
    frame = cv.flip(frame, 1)
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    
    frame_hands = hands.process(frame)

    # Convert back to original colour space for display
    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    cv.imshow('MediaPipe Hands', frame)

    if frame_hands.multi_hand_landmarks: 

        label = ['','']
        label_num = 0
        for i in frame_hands.multi_handedness:
            label[label_num] = MessageToDict(i)['classification'][0]['label']
            label_num += 1
            if label_num > 1:
                continue

        if len(frame_hands.multi_handedness) == 2:
        
            # 0, 2, 4 used for thumb -- 0, 5, 8 used for index -- 0, 9, 12 used for middle -- 0, 13, 16 used for ring -- 0, 17, 20 used for pinky
            return [frame_hands.multi_hand_landmarks[0].landmark[0].x * frame_width,    # 0
                    frame_hands.multi_hand_landmarks[0].landmark[0].y * frame_height,   # 1

                    frame_hands.multi_hand_landmarks[0].landmark[2].x * frame_width,    # 2
                    frame_hands.multi_hand_landmarks[0].landmark[2].y * frame_height,   # 3
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

                    frame_hands.multi_hand_landmarks[1].landmark[2].x * frame_width, 
                    frame_hands.multi_hand_landmarks[1].landmark[2].y * frame_height,
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

                    frame_hands.multi_hand_landmarks[0].landmark[2].x * frame_width, 
                    frame_hands.multi_hand_landmarks[0].landmark[2].y * frame_height,
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
    left_fingers_up  = [False, False, False, False, False]
    right_fingers_up = [False, False, False, False, False]

    if left:
        # TODO: Handle thumbs up case
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

        if l_wrist_index_t > 1.5 * l_wrist_index_k:
            left_fingers_up[1] = True
        if l_wrist_middle_t > 1.5 * l_wrist_middle_k:
            left_fingers_up[2] = True
        if l_wrist_ring_t > 1.5 * l_wrist_ring_k:
            left_fingers_up[3] = True
        if l_wrist_pinky_t > 1.5 * l_wrist_pinky_k:
            left_fingers_up[4] = True
        
    
    if right:
        # TODO: Handle thumbs up case
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

        if r_wrist_index_t > 1.5 * r_wrist_index_k:
            right_fingers_up[1] = True
        if r_wrist_middle_t > 1.5 * r_wrist_middle_k:
            right_fingers_up[2] = True
        if r_wrist_ring_t > 1.5 * r_wrist_ring_k:
            right_fingers_up[3] = True
        if r_wrist_pinky_t > 1.5 * r_wrist_pinky_k:
            right_fingers_up[4] = True

    return left_fingers_up, right_fingers_up


def main():

    hands = mp.solutions.hands

    # Open video capture using webcam (default)
    capture = cv.VideoCapture(0)
    got_frame, frame = capture.read()
    frame_height, frame_width, c = frame.shape 

    mp_hands = mp.solutions.hands

    # Setting up hands model for video capture
    hands = mp_hands.Hands(
        model_complexity = 0,
        max_num_hands = 2,
        min_detection_confidence = 0.5,
        min_tracking_confidence = 0.5
    )

    while True:
        hand1, hand2 = get_position(hands, capture, frame_height, frame_width, mp_hands)
        left_fingers, right_fingers = fingers_up(hand1, hand2)

        print(left_fingers, right_fingers)
        
        if cv.waitKey(1) & 0xff == ord('q'):
            capture.release()

if __name__=="__main__":
    main()
    
