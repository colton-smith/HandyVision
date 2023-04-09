""" hand_pose_engine_test.py

Example usage of hand pose estimation engine (HPEE)
"""
import cv2 as cv

import handyvision as hv


def main():
    """ 
    Print poses of left and right hands 
    using HPEEE.
    """
    hpee = hv.HPEE()
    cam = hv.Camera(0)
    if not cam.good():
        print("Error configuring camera!")
        return

    cam.set_auto_focus(False)
    fps_counter: hv.FPSCounter = hv.FPSCounter()
    while True:
        fps, frametime_ms = fps_counter.update()

        got_frame, frame = cam.get_frame()
        if not got_frame:
            print("Failed to get frame!")
            continue

        # Convert frame to rgb and flip
        frame.flags.writeable = False
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame = cv.flip(frame, 1)
        
        # Update estimation engine
        hpee.update(frame)

        # Get gestures 
        left_g, right_g = hpee.get_gesture_estimation_strings()
        
        # Print gestures
        print(f"FRAME_TIME: {frametime_ms:.2f}ms FPS: {fps:.2f} LEFT: {left_g} | RIGHT: {right_g}")

        # Display frame
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        cv.imshow("Hand Pose Estimation Engine", frame)

        if cv.waitKey(1) & 0xff == ord('q'):
            print("Closing...")
            break

if __name__=="__main__":
    main()
