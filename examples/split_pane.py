import cv2 as cv
import handyvision as hv

def main():
    # Open video capture using webcam (default)
    cam = hv.Camera(0)
    cam.set_auto_focus(False)

    got_frame, frame = cam.get_frame()
    if not got_frame:
        print("Could not get calibration frame, closing...")

    _, frame_width, _ = frame.shape 

    # Set up hand pose estimate engines 
    p1_hpee = hv.HPEE() 
    p2_hpee = hv.HPEE()
    
    while True:
        got_frame, frame = cam.get_frame()        
        if not got_frame:
            print("Missed frame..")
            continue

        # Set frame as read-only to pass by reference
        frame.flags.writeable = False
        frame_p2, frame_p1 = hv.vertically_bisect_image(frame)

        p1_hpee.update(hv.bgr2rgb(frame_p1))
        p2_hpee.update(hv.bgr2rgb(frame_p2))

        p1_left_g, p1_right_g = p1_hpee.get_gesture_estimations()
        p2_left_g, p2_right_g = p2_hpee.get_gesture_estimations()

        p1_left_g_name = hv.get_string_from_gesture(p1_left_g)
        p1_right_g_name = hv.get_string_from_gesture(p1_right_g)

        p2_left_g_name = hv.get_string_from_gesture(p2_left_g)
        p2_right_g_name = hv.get_string_from_gesture(p2_right_g)

        # Any test to be written on the frame should be written after this flip
        frame = cv.flip(frame, 1)
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

        # If player 1 left or right is FLIPOFF
        if  p1_left_g == hv.Gesture.FLIPOFF or p1_right_g == hv.Gesture.FLIPOFF:
            frame[:, frame_width//2:frame_width-1, 2] = 255

        # If player 1 left or right is FLIPOFF
        if  p2_left_g == hv.Gesture.FLIPOFF or p2_right_g == hv.Gesture.FLIPOFF:
            frame[:, 0:frame_width//2, 2] = 255

        cv.imshow("Game Window", frame)
    
        if cv.waitKey(1) & 0xff == ord('q'):
            cam.release()
            break

if __name__=="__main__":
    main()
