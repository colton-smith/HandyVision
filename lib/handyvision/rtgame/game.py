""" game.py

Main game runtime.
"""
import handyvision as hv
import handyvision.rtgame as rtg
import  cv2 as cv
import time


class Game:
    def __init__(self, asset_folder, camera_idx, width = 1280, height=720):
        self.init_success = False
        self.asset_folder = asset_folder
        self.camera_idx = camera_idx
        
        self.cam = hv.Camera(self.camera_idx)
        self.cam.set_auto_focus(False)
        self.cam.set_resolution(width, height)

        got_frame, frame = self.cam.get_frame()
        if not got_frame:
            print("Could not get calibration frame, initialization failed")
            return
        
        self.frame_height, self.frame_width, _ = frame.shape 
        self.init_success = True

        # Set up hand pose estimate engines 
        self.p1_hpee = hv.HPEE() 
        self.p2_hpee = hv.HPEE()

        self.icons = hv.IconManager(self.asset_folder) 

        # Game config
        self.config = rtg.GameConfig()
        self.config.gesture_icon_scale = (0.5, 0.5)
        self.config.flip_off_blur_config.ksize = (7,7)
        self.config.flip_off_blur_config.sigmaX = 5
        self.config.flip_off_blur_config.sigmaY = 5

        # FPS
        self.fps = 0
        self.frametime_ms = 0

        # Flags 
        self.show_fps = False
        self.draw_hand_landmarks = False
        self.show_help = False
        self.show_debug = False

        # Keys
        self.show_fps_key = "f"
        self.draw_hand_landmarks_key = "s"
        self.show_help_key = "h"
        self.show_debug_info_key = "d"

        self.state = rtg.GameState.IDLE
        self.hud = rtg.HUD(self.frame_height, self.frame_width)
        self.fps_counter = hv.FPSCounter()
        self.p1_score = 0
        self.p2_score = 0

    def run(self):
        if not self.init_success:
            print("Not initialized, can not run")
            return
        
        while True:
            self.fps, self.frametime_ms = self.fps_counter.update()
            got_frame, frame = self.cam.get_frame()        
            if not got_frame:
                print("Missed frame..")
                continue

            # Set frame as read-only to pass by reference
            frame = cv.flip(frame, 1)
            frame.flags.writeable = False
            frame_p1, frame_p2, slice_p1, slice_p2 = hv.vertically_bisect_image(frame)

            self.p1_hpee.update(hv.bgr2rgb(frame_p1))
            self.p2_hpee.update(hv.bgr2rgb(frame_p2))

            p1_left_g, p1_right_g = self.p1_hpee.get_gesture_estimations()
            p2_left_g, p2_right_g = self.p2_hpee.get_gesture_estimations()

            frame.flags.writeable = True
            
            # Check both players for FLIPOFF 
            if  p1_left_g == hv.Gesture.FLIPOFF or p1_right_g == hv.Gesture.FLIPOFF:
                frame = self.hud.muddle_frame(frame, hv.HorizontalHalf.RIGHT, self.config)

            if  p2_left_g == hv.Gesture.FLIPOFF or p2_right_g == hv.Gesture.FLIPOFF:
                frame  = self.hud.muddle_frame(frame, hv.HorizontalHalf.LEFT, self.config)

            if self.draw_hand_landmarks:
                frame[slice_p1] = self.p1_hpee.annotate_frame(frame_p1)
                frame[slice_p2] = self.p2_hpee.annotate_frame(frame_p2)

            # Draw all consistent UI
            frame = self.hud.draw_hud(frame)
            self.hud.draw_player_scores(frame, self.p1_score, self.p2_score)

            if self.show_debug and self.state != rtg.GameState.COUNTDOWN:
                self.hud.draw_pose_estimates(frame, (p1_left_g, p1_right_g), (p2_left_g, p2_right_g))

            if self.show_fps: 
                self.hud.draw_fps(frame, self.fps)

            # Draw help on top of other hid items
            if self.show_help:
                self.hud.draw_help_box(frame)
        
            # TODO: Can crash
            match self.state:
                case rtg.GameState.IDLE:
                    self.hud.draw_idle_text(frame)
                    self.p1_score = 0
                    self.p2_score = 0
                    p1_winner = False
                    p2_winner = False
                    # TODO: This crashes if hands are in frame with the same
                    # pose at the start, start_time ref before assignment
                    if not (p1_left_g == p2_right_g != hv.Gesture.OUT_OF_FRAME):
                        start_time = time.time()
                    elif p1_left_g == p2_right_g != hv.Gesture.OUT_OF_FRAME:
                        current_time = time.time()
                        if current_time - start_time >= 2:
                            if p1_left_g not in self.config.gest_to_rounds:
                                number_rounds = self.config.default_num_rounds
                            else:
                                number_rounds = self.config.gest_to_rounds[p1_left_g]
                            countdown_start_time = current_time
                            self.state = rtg.GameState.COUNTDOWN

                case rtg.GameState.COUNTDOWN:
                    count = self.config.start_game_count - int(time.time() - countdown_start_time)
                    if count <= 0:
                        self.state = rtg.GameState.DISPLAY_GESTURE
                    else:
                        frame = self.hud.draw_countdown_page(frame, number_rounds, count)

                case rtg.GameState.DISPLAY_GESTURE:
                    target_gest_left = hv.get_random_gesture([hv.Gesture.FLIPOFF, hv.Gesture.OUT_OF_FRAME])
                    target_gest_right = hv.get_random_gesture([hv.Gesture.FLIPOFF, hv.Gesture.OUT_OF_FRAME])
                    
                    left_icon = self.icons.icon_for_gesture(
                        hv.Handedness.LEFT, 
                        target_gest_left, 
                        self.config.gesture_icon_scale
                    )
                    right_icon = self.icons.icon_for_gesture(
                        hv.Handedness.RIGHT, 
                        target_gest_right, 
                        self.config.gesture_icon_scale
                    )
                    self.state = rtg.GameState.CHECK_GESTURE

                case rtg.GameState.CHECK_GESTURE:
                    frame = self.hud.draw_gesture_icons(frame, left_icon, right_icon)
                    
                    p1_correct = p1_left_g == target_gest_left and p1_right_g == target_gest_right
                    p2_correct = p2_left_g == target_gest_left and p2_right_g == target_gest_right

                    if p1_correct and p2_correct:
                        print("TIE")
                        frame[:, :, 0:1] = 150
                        countdown_start_time = time.time()
                        self.state = rtg.GameState.DISPLAY_GESTURE
                    elif p1_correct:
                        self.p1_score += 1
                        frame[:, 0:self.frame_width//2, 1] = 200
                        countdown_start_time = time.time()
                        if self.p1_score == number_rounds:
                            p1_winner = True
                            self.state = rtg.GameState.WIN_SCREEN
                            winscreen_starttime = time.time()
                        elif self.p2_score == number_rounds:
                            p2_winner = True
                            self.state = rtg.GameState.WIN_SCREEN
                            winscreen_starttime = time.time()
                        else:
                            self.state = rtg.GameState.DISPLAY_GESTURE
                    elif p2_correct:
                        self.p2_score += 1
                        frame[:, self.frame_width//2:self.frame_width-1, 1] = 200
                        countdown_start_time = time.time()
                        if self.p1_score == number_rounds:
                            p1_winner = True
                            self.state = rtg.GameState.WIN_SCREEN
                            winscreen_starttime = time.time()
                        elif self.p2_score == number_rounds:
                            p2_winner = True
                            self.state = rtg.GameState.WIN_SCREEN
                            winscreen_starttime = time.time()
                        else:
                            self.state = rtg.GameState.DISPLAY_GESTURE

                case rtg.GameState.WIN_SCREEN:
                    winscreen_currenttime = time.time()
                    if p1_winner:
                        # Disable green screen
                        # frame[:, 0:self.frame_width//2, 1] = 200
                        self.hud.draw_player_wins_text(frame, "1")
                    elif p2_winner:
                        # frame[:, self.frame_width//2:self.frame_width-1, 1] = 200
                        self.hud.draw_player_wins_text(frame, "2")
                    
                    if winscreen_currenttime - winscreen_starttime > 5:
                        self.state = rtg.GameState.IDLE
            
            cv.imshow("HandyVision!", frame)

            key = cv.waitKey(1)
            if key & 0xff == ord(self.show_debug_info_key):
                print("Toggling show debug!")
                self.show_debug  = not self.show_debug
            if key & 0xff == ord(self.show_help_key):
                print("Toggling show help!")
                self.show_help = not self.show_help
            if key & 0xff == ord(self.draw_hand_landmarks_key):
                print("Toggling show landmarks!")
                self.draw_hand_landmarks = not self.draw_hand_landmarks
            if key & 0xff == ord(self.show_fps_key):
                print("Toggling show fps!")
                self.show_fps = not self.show_fps
            if key & 0xff == ord('q'):
                self.cam.release()
                break

    