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


class LineOptions: 
    def __init__(self, color = None, thickness = None):
        default_color = (255, 255, 255)
        default_thickness = 4

        self.color = color if color is not None else default_color 
        self.thickness = thickness if thickness is not None else default_thickness 


class GameConfig:
    def __init__(self):
        self.p1_label_color = (255, 255, 255)
        self.p2_label_color = (255, 255, 255)
        self.gesture_icon_size = (100, 100)
        self.skeleton_toggle_key = 's'
        self.fps_count_toggle_key= 'f'
        self.draw_center_line = False


def draw_gesture_icons(frame: cv.Mat, left: cv.Mat, right: cv.Mat) -> cv.Mat:
    """ Draw left and right gesture icons on the frame
    """
    # TODO
    center_col = frame.shape[1] // 2
    img_w = left.shape[1]
    img_h = left.shape[0]

    # Icon size (H/W)
    icon_size = np.array([img_h, img_w]) 

    # Vertical bumper from top of screen
    vertical_padding = 10

    start_col = center_col - img_w
    start_row = vertical_padding

    left_image_top_left = np.array([start_row, start_col])
    left_image_bot_right = left_image_top_left + icon_size 

    frame[left_image_top_left[0]:left_image_bot_right[0], 
          left_image_top_left[1]:left_image_bot_right[1],:] = left
    
    # TODO: Need to handle transparency
    return frame


def draw_center_line(frame: cv.Mat, options: LineOptions) -> cv.Mat:
    """ Draw center line on frame with options
    """
    h = frame.shape[0]
    w = frame.shape[1]
    frame = cv.line(frame, (w//2, 0), (w//2, h-1), options.color, options.thickness)
    return frame


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
    
    # Configure icon manager 
    icon_manager = hv.IconManager("C:/Users/cnsmith/dev/HandyVision/assets") 

    # Game config
    config = GameConfig()
    config.draw_center_line = True

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

        font = cv.FONT_HERSHEY_SIMPLEX
        frame.flags.writeable = True

        if config.draw_center_line:
            frame = draw_center_line(frame, LineOptions()) 
        
        # Check both players for FLIPOFF 
        if  p1_left_g == hv.Gesture.FLIPOFF or p1_right_g == hv.Gesture.FLIPOFF:
            frame[:, frame_width//2:frame_width-1, 2] = 255

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
    
        # Crashes
        match state:
            case GameState.IDLE:
                p1_score = 0
                p2_score = 0
                p1_winner = False
                p2_winner = False
                # This crashes if hands are in frame with the same
                # pose at the start, start_time ref before assignment
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
                
                left_icon = icon_manager.icon_for_gesture(hv.Handedness.LEFT, compare_gesture_l)
                right_icon = icon_manager.icon_for_gesture(hv.Handedness.RIGHT, compare_gesture_r)

                frame = cv.putText(frame, "Left: " + compare_gesture_l.name, coordinates_gesture_l, font, 1, (0,255,0), 3, cv.LINE_AA)
                frame = cv.putText(frame, "Right: " + compare_gesture_r.name, coordinates_gesture_r, font, 1, (0,255,0), 3, cv.LINE_AA)
                state = GameState.CHECK_GESTURE

            case GameState.CHECK_GESTURE:
                frame = draw_gesture_icons(frame, left_icon, right_icon)

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
