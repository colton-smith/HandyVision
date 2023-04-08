""" camera_test.py

Example usage of handyvision.Camera 
"""
import handyvision as hv
import cv2 as cv

def main():
    cam: hv.Camera = hv.Camera(0, "Webcam")
    
    # Check if camera is good
    if cam.good():
        print("Camera good")
    else:
        print("Unable to open camera")
        return
    
    while True:
        # Collect and display frames 
        got_frame, frame = cam.get_frame()

        if not got_frame:
            print("Could not get frame")
        else:
            cv.imshow("Camera Frame", frame)

        # Wait on input
        if cv.waitKey(1) & 0xff == ord('q'):
            cam.release()
            break
    

if __name__ == "__main__":
    """ Camera test
    """
    main()
