""" hud.py

Heads up display drawing utilities. 
"""
import cv2 as cv
import numpy as np
import handyvision as hv
import handyvision.rtgame as rtg
import textwrap


class HUD:
    """ Track and draw game HUD elements on camera frame
    """
    def __init__(self, frame_height, frame_width):

        self.frame_height = frame_height
        self.frame_width = frame_width

        self.player1_str = "P1"
        self.player2_str = "P2"

        self.primary_hud_color = hv.COLORS["SLATE_GREY"]
        self.hud_text_color = hv.COLORS["WHITE"]

        self.fps_font = hv.Font(
            cv.FONT_HERSHEY_COMPLEX,
            0.75,
            2,
            self.hud_text_color
        )

        self.player_label_font = hv.Font(
            cv.FONT_HERSHEY_COMPLEX,
            0.75,
            2,
            self.hud_text_color
        )

        self.countdown_text_font = hv.Font(
            cv.FONT_HERSHEY_COMPLEX,
            0.75,
            2,
            self.hud_text_color
        )
        
        self.countdown_number_font = hv.Font(
            cv.FONT_HERSHEY_COMPLEX,
            4,
            5,
            self.hud_text_color
        )

        self.idle_text_font = hv.Font(
            cv.FONT_HERSHEY_COMPLEX,
            1,
            2,
            self.hud_text_color
        )

        self.idle_subtext_font = hv.Font(
            cv.FONT_HERSHEY_COMPLEX,
            0.75,
            2,
            self.hud_text_color
        )

        self.help_text_font = hv.Font(
            cv.FONT_HERSHEY_COMPLEX_SMALL,
            0.65,
            1,
            self.hud_text_color
        )

        self.hud_box = hv.Rect(
            width=300,
            height=175,
            color=self.primary_hud_color,
            thickness=-1,
            line_type=cv.LINE_AA
        )

        self.vertical_divider = hv.Line(
            color=self.primary_hud_color,
            thickness=34
        )

        self.bottom_rect = hv.Rect(
            width=frame_width,
            height=34,
            color=self.primary_hud_color,
            thickness=-1,
            line_type=cv.LINE_AA
        )

        self.top_left_rect = hv.Rect(
            width=200,
            height=34,
            color=self.primary_hud_color,
            thickness=-1,
            line_type=cv.LINE_AA
        )

        self.top_right_rect = hv.Rect(
            width=200,
            height=34,
            color=self.primary_hud_color,
            thickness=-1,
            line_type=cv.LINE_AA
        )

    def draw_hud(self, frame: cv.Mat):
        """ Draw persistent HUD
        """
        self.draw_hud_box(frame)
        self.draw_vertical_divider(frame)
        self.draw_bottom_bar(frame)
        self.draw_top_bars(frame)
        self.draw_player_labels(frame)
        return frame

    def get_hud_box_top_left(self, frame: cv.Mat):
        """ Get top left of hux box w.r.t frame
        """
        fh, fw, _ = frame.shape
        center_col = fw // 2
        top_left = center_col - self.hud_box.width // 2, 0
        return top_left
    
    def get_hud_box_centre(self, frame):
        """ Get center of hud box
        """
        _, fw, _ = frame.shape
        center_col = fw // 2
        return center_col, self.hud_box.height // 2

    def draw_hud_box(self, frame: cv.Mat):
        """ Draw hud box in upper center of screen
        """
        return hv.draw_rectangle(frame, self.hud_box, self.get_hud_box_top_left(frame))
    
    def draw_vertical_divider(self, frame: cv.Mat):
        """ Draw divider
        """
        return hv.draw_center_line(frame, self.vertical_divider) 

    def draw_bottom_bar(self, frame: cv.Mat):
        """ Draw bottom bar
        """
        h, _, _ = frame.shape
        return hv.draw_rectangle(frame, self.bottom_rect, (0, h - self.bottom_rect.height))

    def draw_top_bars(self, frame: cv.Mat):
        """ Draw left and right top bars
        """
        _, w, _ = frame.shape
        rect_w = self.top_right_rect.width
        hv.draw_rectangle(frame, self.top_right_rect, (w - rect_w, 0))
        hv.draw_rectangle(frame, self.top_left_rect, (0, 0))
        
    def draw_player_labels(self, frame: cv.Mat):
        """ Draw player labels
        """
        hv.draw_text_top_left(frame, self.player1_str, self.player_label_font)
        hv.draw_text_top_right(frame, self.player2_str, self.player_label_font)

    def draw_player_scores(self, frame: cv.Mat, p1_score, p2_score) -> cv.Mat:
        """ Draw player scores
        """
        # P1
        padding = 10
        size, baseline = hv.get_text_size(self.player1_str, self.player_label_font)
        p1_left = size[0] + padding
        p1_bot = size[1] + baseline

        hv.draw_text(frame, f" | {p1_score}", self.player_label_font, (p1_left, p1_bot))

        # P2 
        size, baseline = hv.get_text_size(self.player2_str, self.player_label_font)
        score_size, _ = hv.get_text_size(f"{p2_score} | ", self.player_label_font)

        p2_left = frame.shape[1] - size[0] - score_size[0] - padding
        p2_bot = size[1] + baseline
        hv.draw_text(frame, f"{p2_score} | ", self.player_label_font, (p2_left, p2_bot))

    def draw_help_box(self, frame: cv.Mat):
        """ Draw big old help box explaining the app usage
        """
        # Draw big ass rectanle
        # Draw text inside
        h, w, _ = frame.shape
        rect = hv.Rect(
            600,
            400,
            self.primary_hud_color,
            -1,
            cv.LINE_AA
        )

        top_left = w // 2 - rect.width // 2, h // 2 - rect.height // 2
        hv.draw_rectangle(frame, rect, top_left)

        # Draw text one line at a 
        lines = [
            "Welcome to Handy Vision!",
            "To start:",
            "P1: Make a peace sign with left hand, hold",
            "P2: Make a peace sign with right hand, hold",
            "After some time, the countdown and match will start",
            "Using other hand signs changes the # of rounds",
            "Attempt to match the hand signs shown to get a point!",
            "The first player to reach the target number of matches wins!",
            "Controls:",
            "D - Toggle pose debug",
            "S - Toggle hand skeletons",
            "F - Toggle frame rate counter",
            "Q - Quit"
        ]

        padding = 24
        line_left = top_left[0] + 24
        line_bot = top_left[1] + 35
        for line in lines:
            hv.draw_text(frame, line, self.help_text_font, (line_left, line_bot))
            line_bot += padding

    def draw_player_wins_text(self, frame: cv.Mat, winner):
        """ Draw winner message in HUD box
        """
        text = f"Player {winner} Wins!"
        self.draw_text_centered_in_hud_box(frame, text, self.idle_text_font)

    def draw_idle_text(self, frame: cv.Mat):
        """ Draw help text in the HUD box
        """
        text = f"Handy Vision!"
        self.draw_text_centered_in_hud_box(frame, text, self.idle_text_font)

        padding = 40
        sub_text = f"H for Help"
        centre_hud = self.get_hud_box_centre(frame)
        centre = centre_hud[0], centre_hud[1] + padding
        size, baseline = hv.get_text_size(sub_text, self.idle_subtext_font)
        bot_left = hv.get_text_bot_left(size, baseline, centre)
        hv.draw_text(frame, sub_text, self.idle_subtext_font, bot_left)

    def draw_fps(self, frame: cv.Mat, fps: float) -> cv.Mat:
        """ Draw FPS
        """
        fps_string = f"{fps:0.0f}"
        return hv.draw_text_bottom_right(frame, fps_string, self.fps_font)
        
    def draw_countdown_page(self, frame: cv.Mat, rounds: int, count: int) -> cv.Mat:
        """ Draw countdown page on frame
        """
        countdown_display_str = f"Starting game with {rounds} rounds..."
        frame, _ = hv.draw_text_bottom_left(frame, countdown_display_str, self.countdown_text_font)
        frame = self.draw_text_centered_in_hud_box(frame, f"{count}", self.countdown_number_font)
        return frame

    def draw_text_centered_in_hud_box(self, frame: cv.Mat, text: str, font: hv.Font):
        """ Draw text at center of hud box
        """
        hud_centre = self.get_hud_box_centre(frame)
        text_size, baseline = hv.get_text_size(text, font)
        text_bot_left = hv.get_text_bot_left(text_size, baseline, hud_centre)
        return hv.draw_text(frame, text, font, text_bot_left)

    def draw_pose_estimates(self, frame, p1_gs, p2_gs):
        """ Draw the current pose estimates (left gesture, right gesture)
        """
        p1_l, p1_r = hv.get_string_from_gesture(p1_gs[0]), hv.get_string_from_gesture(p1_gs[1])
        p1_str = f"L: {p1_l} | R: {p1_r}"

        h, w, _ = frame.shape
        p1_size, p1_baseline = hv.get_text_size(p1_str, self.fps_font)
        bot_left = w // 4 - p1_size[0] // 2, h - p1_size[1] - 1 + p1_baseline
        hv.draw_text(frame, p1_str, self.fps_font, bot_left)

        p2_l, p2_r = hv.get_string_from_gesture(p2_gs[0]), hv.get_string_from_gesture(p2_gs[1])
        p2_str = f"L: {p2_l} | R: {p2_r}"
        p2_size, p2_baseline = hv.get_text_size(p2_str, self.fps_font)
        bot_left = 3 * w // 4 - p1_size[0] // 2, h - p1_size[1] - 1 + p1_baseline
        hv.draw_text(frame, p2_str, self.fps_font, bot_left)

    def draw_gesture_icons(self, frame: cv.Mat, left: cv.Mat, right: cv.Mat) -> cv.Mat:
        """ Draw left and right gesture icons on the frame
        """
        center_col = frame.shape[1] // 2
        overlay_w = left.shape[1]
        inter_icon_spacer_half_width = 5

        # Vertical bumper from top of screen
        vertical_padding = 10

        # Offset from center width of overlay + half the spacer width
        start_col = center_col - overlay_w - inter_icon_spacer_half_width
        start_row = vertical_padding

        left_image_top_left = np.array([start_row, start_col])
        frame = hv.overlay_transparent_image(frame, left, left_image_top_left)

        # Offset right image width of the left overlay, + 2 * spacer
        right_image_top_left = left_image_top_left + (0, overlay_w + 2 * inter_icon_spacer_half_width)
        frame = hv.overlay_transparent_image(frame, right, right_image_top_left)

        return frame

    def muddle_frame(self, frame: cv.Mat, half: hv.HorizontalHalf, config: rtg.GameConfig):
        """ Muddle one half of the frame 

        Applies a gaussian filter and tints the screen frame red
        """
        _, cols, _ = frame.shape

        # Tint + Blur
        blur_options =config.flip_off_blur_config
        ksize = blur_options.ksize
        sx = blur_options.sigmaX
        sy = blur_options.sigmaY

        if half == hv.HorizontalHalf.LEFT:
            frame[:, 0:cols//2, 2] = 255
            frame[:, 0:cols//2, :] = cv.GaussianBlur(frame[:, 0:cols//2, :], ksize, sx, sy)
        else:
            frame[:, cols//2:cols, 2] = 255
            frame[:, cols//2:cols, :] = cv.GaussianBlur(frame[:, cols//2:cols, :], ksize, sx, sy)

        return frame
