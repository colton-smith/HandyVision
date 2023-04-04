import cv2 as cv
import mediapipe as mp
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

        if len(frame_hands.multi_handedness) == 2:
        
            return [frame_hands.multi_hand_landmarks[0].landmark[0].x * frame_width, 
                    frame_hands.multi_hand_landmarks[0].landmark[0].y * frame_height,
                    frame_hands.multi_hand_landmarks[0].landmark[4].x * frame_width, 
                    frame_hands.multi_hand_landmarks[0].landmark[4].y * frame_height,
                    frame_hands.multi_hand_landmarks[0].landmark[8].x * frame_width, 
                    frame_hands.multi_hand_landmarks[0].landmark[8].y * frame_height,
                    frame_hands.multi_hand_landmarks[0].landmark[12].x * frame_width,
                    frame_hands.multi_hand_landmarks[0].landmark[12].y * frame_height,
                    frame_hands.multi_hand_landmarks[0].landmark[16].x * frame_width, 
                    frame_hands.multi_hand_landmarks[0].landmark[16].y * frame_height,
                    frame_hands.multi_hand_landmarks[0].landmark[20].x * frame_width, 
                    frame_hands.multi_hand_landmarks[0].landmark[20].y * frame_height,
                    label[0]], \
                   [frame_hands.multi_hand_landmarks[1].landmark[0].x * frame_width, 
                    frame_hands.multi_hand_landmarks[1].landmark[0].y * frame_height,
                    frame_hands.multi_hand_landmarks[1].landmark[4].x * frame_width, 
                    frame_hands.multi_hand_landmarks[1].landmark[4].y * frame_height,
                    frame_hands.multi_hand_landmarks[1].landmark[8].x * frame_width, 
                    frame_hands.multi_hand_landmarks[1].landmark[8].y * frame_height,
                    frame_hands.multi_hand_landmarks[1].landmark[12].x * frame_width,
                    frame_hands.multi_hand_landmarks[1].landmark[12].y * frame_height,
                    frame_hands.multi_hand_landmarks[1].landmark[16].x * frame_width, 
                    frame_hands.multi_hand_landmarks[1].landmark[16].y * frame_height,
                    frame_hands.multi_hand_landmarks[1].landmark[20].x * frame_width, 
                    frame_hands.multi_hand_landmarks[1].landmark[20].y * frame_height,
                    label[1]]
        
        else:
            return [frame_hands.multi_hand_landmarks[0].landmark[0].x * frame_width, 
                    frame_hands.multi_hand_landmarks[0].landmark[0].y * frame_height,
                    frame_hands.multi_hand_landmarks[0].landmark[4].x * frame_width, 
                    frame_hands.multi_hand_landmarks[0].landmark[4].y * frame_height,
                    frame_hands.multi_hand_landmarks[0].landmark[8].x * frame_width, 
                    frame_hands.multi_hand_landmarks[0].landmark[8].y * frame_height,
                    frame_hands.multi_hand_landmarks[0].landmark[12].x * frame_width,
                    frame_hands.multi_hand_landmarks[0].landmark[12].y * frame_height,
                    frame_hands.multi_hand_landmarks[0].landmark[16].x * frame_width, 
                    frame_hands.multi_hand_landmarks[0].landmark[16].y * frame_height,
                    frame_hands.multi_hand_landmarks[0].landmark[20].x * frame_width, 
                    frame_hands.multi_hand_landmarks[0].landmark[20].y * frame_height,
                    label[0]], [0,0,0,0,0,0,0,0,0,0,0,0,0]
               
    return [0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0]

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
        print(hand1)
        print(hand2, '\n')
        
        if cv.waitKey(1) & 0xff == ord('q'):
            capture.release()

if __name__=="__main__":
    main()
    
