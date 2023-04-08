import cv2 as cv
import mediapipe as mp
import handyvision as hv

def process_frame(hand_model, frame: cv.Mat):
    """ Pass frame through mediapipe and play with results 
    """
    frame = cv.flip(frame, 1)
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame_hands = hand_model.process(frame)

    # Just grab the first hand
    if frame_hands is None:
        return 
    
    if frame_hands.multi_hand_landmarks is None:
        return 
    
    # Collect the first hand if there is one 
    mp_landmarks = frame_hands.multi_hand_landmarks[0].landmark
    landmarks = hv.HandLandmarks(mp_landmarks)

    landmark_ids = [e for e in hv.HandLandmarkName]
    print("--- HAND 0 LANDMARK READINGS --- ")
    for id in landmark_ids:
        print(f"{id.name}: {landmarks.get(id)}")
    print("---") 


def main():

    # Open video capture using webcam (default)
    cam = hv.Camera(0)
  
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
        frame_height, frame_width, c = frame.shape 
        process_frame(hands, frame)

        # Display frame 
        cv.imshow("Hands", frame)
        if cv.waitKey(1) & 0xff == ord('q'):
            break

if __name__=="__main__":
    main()
    