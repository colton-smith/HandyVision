""" reaction_game.py

Reaction time game.
"""
import cv2 as cv
import handyvision as hv
import time
import numpy as np
from enum import IntEnum, auto, unique


@unique 
class GameState(IntEnum):
    """ Game state enum
    """
    IDLE = 0
    COUNTDOWN = auto()
    DISPLAY_GESTURE = auto()
    CHECK_GESTURE = auto()
    WIN_SCREEN = auto()


def main():
    # Open video capture using webcam (default)
    cam = hv.Camera(0)
    cam.set_auto_focus(False)

    got_frame, frame = cam.get_frame()
    if not got_frame:
        print("Could not get calibration frame, closing...")

    frame_height, frame_width, _ = frame.shape 

    # Set up hand pose estimate engines 
    p1_hpee = hv.HPEE() 
    p2_hpee = hv.HPEE()
    
    state: GameState = GameState.IDLE
    
    while True:
        got_frame, frame = cam.get_frame()        
        if not got_frame:
            print("Missed frame..")
            continue

        # Set frame as read-only to pass by reference
        frame = cv.flip(frame, 1)
        frame.flags.writeable = False
        frame_p1, frame_p2 = hv.vertically_bisect_image(frame)

        p1_hpee.update(hv.bgr2rgb(frame_p1))
        p2_hpee.update(hv.bgr2rgb(frame_p2))

        p1_left_g, p1_right_g = p1_hpee.get_gesture_estimations()
        p2_left_g, p2_right_g = p2_hpee.get_gesture_estimations()

        p1_left_g_name = hv.get_string_from_gesture(p1_left_g)
        p1_right_g_name = hv.get_string_from_gesture(p1_right_g)

        p2_left_g_name = hv.get_string_from_gesture(p2_left_g)
        p2_right_g_name = hv.get_string_from_gesture(p2_right_g)

        # Any test to be written on the frame should be written after this flip
    
        coordinates_l_p1 = (50, 50)
        coordinates_r_p1 = (50, 100)
        coordinates_l_p2 = (50 + frame_width // 2, 50)
        coordinates_r_p2 = (50 + frame_width // 2, 100)
        font = cv.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (255, 0 , 255)
        thickness = 2
        frame.flags.writeable = True
        frame = cv.putText(frame, p1_left_g_name, coordinates_l_p1, font, fontScale, color, thickness, cv.LINE_AA)
        frame = cv.putText(frame, p1_right_g_name, coordinates_r_p1, font, fontScale, color, thickness, cv.LINE_AA)
        frame = cv.putText(frame, p2_left_g_name, coordinates_l_p2, font, fontScale, color, thickness, cv.LINE_AA)
        frame = cv.putText(frame, p2_right_g_name, coordinates_r_p2, font, fontScale, color, thickness, cv.LINE_AA)
        frame = cv.line(frame, (frame_width//2, 0), (frame_width//2, frame_height-1), (0,0,0), 10) 
        
        # If player 1 left or right is FLIPOFF
        if  p1_left_g == hv.Gesture.FLIPOFF or p1_right_g == hv.Gesture.FLIPOFF:
            frame[:, frame_width//2:frame_width-1, 2] = 255

        # If player 1 left or right is FLIPOFF
        if  p2_left_g == hv.Gesture.FLIPOFF or p2_right_g == hv.Gesture.FLIPOFF:
            frame[:, 0:frame_width//2, 2] = 255

        # Game State Machine
        # 1 --> Wait until same gestures are held by all 4 hands at the same time for 2 seconds
        #       - If 'point' is displayed, game to 4
        #       - If 'peace' is displayed, game to 8
        #       - If 'three' is displayed, game to 12
        #       - If 'four'  is displayed, game to 16
        #       - If 'spread' is displayed, game to 20
        # 2 --> Start countdown
        # 3 --> Display symbols (write name of left and right gesture) and detect who displayed first
        #       - Add point to winner, check if score is equal to win_number
        #       - if points = win_number --> Win screen
        #       - if points != win_number --> Start countdown
        # p1_left_g_name, gesture_name_r_p1
        # gesture_name_l_p2, p2_right_g_name
    
        match state:
            case GameState.IDLE:
                p1_score = 0
                p2_score = 0
                p1_winner = False
                p2_winner = False
                if not (p1_left_g == p2_right_g != hv.Gesture.OUT_OF_FRAME):
                    start_time = time.time()
                elif p1_left_g == p2_right_g != hv.Gesture.OUT_OF_FRAME:
                    current_time = time.time()
                    if current_time - start_time >= 2:
                        if p1_left_g == hv.Gesture.POINT : number_rounds = 4
                        elif p1_left_g == hv.Gesture.PEACE : number_rounds = 8
                        elif p1_left_g == hv.Gesture.THREE : number_rounds = 12
                        elif p1_left_g == hv.Gesture.FOUR : number_rounds = 16
                        elif p1_left_g == hv.Gesture.SPREAD : number_rounds = 20
                        else : number_rounds = 4    # Default number of rounds
                        countdown_start_time = current_time
                        state = GameState.COUNTDOWN

            case GameState.COUNTDOWN:
                coordinates_count = (frame_width//2 - 50, 200)
                coordinates_gesture_l = (frame_width//2 - 50, 400)
                coordinates_gesture_r = (frame_width//2 - 50, 500)
                count = int(time.time() - countdown_start_time)
                if count <= 2:
                    frame = cv.putText(frame, str(count+1), coordinates_count, font, 5, (0,0,255), 3, cv.LINE_AA)
                if count >= 3:
                    state = GameState.DISPLAY_GESTURE

            case GameState.DISPLAY_GESTURE:
                compare_gesture_l = hv.get_random_gesture()
                compare_gesture_r = hv.get_random_gesture()
                frame = cv.putText(frame, "Left: " + compare_gesture_l.name, coordinates_gesture_l, font, 1, (0,255,0), 3, cv.LINE_AA)
                frame = cv.putText(frame, "Right: " + compare_gesture_r.name, coordinates_gesture_r, font, 1, (0,255,0), 3, cv.LINE_AA)
                state = GameState.CHECK_GESTURE

            case GameState.CHECK_GESTURE:
                frame = cv.putText(frame, "Left: " + compare_gesture_l.name, coordinates_gesture_l, font, 1, (0,0,255), 3, cv.LINE_AA)
                frame = cv.putText(frame, "Right: " + compare_gesture_r.name, coordinates_gesture_r, font, 1, (0,0,255), 3, cv.LINE_AA)
                if p1_left_g == p2_left_g == compare_gesture_l and \
                   p1_right_g == p2_right_g == compare_gesture_r:
                    print("TIE")
                    frame[:, :, 0:1] = 150
                    countdown_start_time = time.time()
                    state = GameState.DISPLAY_GESTURE
                elif p1_left_g == compare_gesture_l and p2_right_g == compare_gesture_r:
                    p1_score += 1
                    frame[:, 0:frame_width//2, 1] = 200
                    countdown_start_time = time.time()
                    if p1_score == number_rounds:
                        p1_winner = True
                        state = GameState.WIN_SCREEN
                        winscreen_starttime = time.time()
                    elif p2_score == number_rounds:
                        p2_winner = True
                        state = GameState.WIN_SCREEN
                        winscreen_starttime = time.time()
                    else:
                        state = GameState.DISPLAY_GESTURE
                elif p2_left_g == compare_gesture_l and p2_right_g == compare_gesture_r:
                    p2_score += 1
                    frame[:, frame_width//2:frame_width-1, 1] = 200
                    countdown_start_time = time.time()
                    if p1_score == number_rounds:
                        p1_winner = True
                        state = GameState.WIN_SCREEN
                        winscreen_starttime = time.time()
                    elif p2_score == number_rounds:
                        p2_winner = True
                        state = GameState.WIN_SCREEN
                        winscreen_starttime = time.time()
                    else:
                        state = GameState.DISPLAY_GESTURE

            case GameState.WIN_SCREEN:
                winscreen_currenttime = time.time()
                if p1_winner:
                    frame[:, 0:frame_width//2, 1] = 200
                elif p2_winner:
                    frame[:, frame_width//2:frame_width-1, 1] = 200
                
                if winscreen_currenttime - winscreen_starttime > 5:
                    state = GameState.IDLE
        
        cv.imshow("Game Window", frame)

        if cv.waitKey(1) & 0xff == ord('q'):
            cam.release()
            break

if __name__=="__main__":
    main()
