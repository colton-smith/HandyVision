""" game.py

Main game runtime.
"""
import handyvision as hv
import handyvision.rtgame as rtg
import  cv2 as cv
import time


class Game:
    def __init__(self, asset_folder, camera_idx):
        self.init_success = False
        self.asset_folder = asset_folder
        self.camera_idx = camera_idx
        
        self.cam = hv.Camera(self.camera_idx)
        self.cam.set_auto_focus(False)

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
        self.config.draw_center_line = False
        self.config.gesture_icon_scale = (0.5, 0.5)
        self.config.flip_off_blur_config.ksize = (7,7)
        self.config.flip_off_blur_config.sigmaX = 5
        self.config.flip_off_blur_config.sigmaY = 5

        # Configuration
        self.show_fps_key = 'f'
        self.show_fps = False
        self.fps = 0
        self.frametime_ms = 0
        

        self.state = rtg.GameState.IDLE
        self.hud = rtg.HUD(self.frame_height, self.frame_width)
        self.fps_counter = hv.FPSCounter()

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
            frame_p1, frame_p2 = hv.vertically_bisect_image(frame)

            self.p1_hpee.update(hv.bgr2rgb(frame_p1))
            self.p2_hpee.update(hv.bgr2rgb(frame_p2))

            p1_left_g, p1_right_g = self.p1_hpee.get_gesture_estimations()
            p2_left_g, p2_right_g = self.p2_hpee.get_gesture_estimations()

            frame.flags.writeable = True
            if self.config.draw_center_line:
                frame = hv.draw_center_line(frame, hv.LineOptions()) 
            
            # Check both players for FLIPOFF 
            if  p1_left_g == hv.Gesture.FLIPOFF or p1_right_g == hv.Gesture.FLIPOFF:
                frame = rtg.muddle_frame(frame, hv.HorizontalHalf.RIGHT, self.config)

            if  p2_left_g == hv.Gesture.FLIPOFF or p2_right_g == hv.Gesture.FLIPOFF:
                frame  = rtg.muddle_frame(frame, hv.HorizontalHalf.LEFT, self.config)

            # Draw all consistent UI
            if self.show_fps: 
                self.hud.draw_fps(frame, self.fps)

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
        
            # TODO: Can crash
            match self.state:
                case rtg.GameState.IDLE:
                    p1_score = 0
                    p2_score = 0
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
                    frame = rtg.draw_gesture_icons(frame, left_icon, right_icon)
                    
                    p1_correct = p1_left_g == target_gest_left and p1_right_g == target_gest_right
                    p2_correct = p2_left_g == target_gest_left and p2_right_g == target_gest_right

                    if p1_correct and p2_correct:
                        print("TIE")
                        frame[:, :, 0:1] = 150
                        countdown_start_time = time.time()
                        self.state = rtg.GameState.DISPLAY_GESTURE
                    elif p1_correct:
                        p1_score += 1
                        frame[:, 0:self.frame_width//2, 1] = 200
                        countdown_start_time = time.time()
                        if p1_score == number_rounds:
                            p1_winner = True
                            self.state = rtg.GameState.WIN_SCREEN
                            winscreen_starttime = time.time()
                        elif p2_score == number_rounds:
                            p2_winner = True
                            self.state = rtg.GameState.WIN_SCREEN
                            winscreen_starttime = time.time()
                        else:
                            self.state = rtg.GameState.DISPLAY_GESTURE
                    elif p2_correct:
                        p2_score += 1
                        frame[:, self.frame_width//2:self.frame_width-1, 1] = 200
                        countdown_start_time = time.time()
                        if p1_score == number_rounds:
                            p1_winner = True
                            self.state = rtg.GameState.WIN_SCREEN
                            winscreen_starttime = time.time()
                        elif p2_score == number_rounds:
                            p2_winner = True
                            self.state = rtg.GameState.WIN_SCREEN
                            winscreen_starttime = time.time()
                        else:
                            self.state = rtg.GameState.DISPLAY_GESTURE

                case rtg.GameState.WIN_SCREEN:
                    winscreen_currenttime = time.time()
                    if p1_winner:
                        frame[:, 0:self.frame_width//2, 1] = 200
                    elif p2_winner:
                        frame[:, self.frame_width//2:self.frame_width-1, 1] = 200
                    
                    if winscreen_currenttime - winscreen_starttime > 5:
                        self.state = rtg.GameState.IDLE
            
            cv.imshow("Game Window", frame)

            key = cv.waitKey(1)

            if key & 0xff == ord(self.show_fps_key):
                print("Toggling show fps!")
                self.show_fps = not self.show_fps

            if key & 0xff == ord('q'):
                self.cam.release()
                break

    