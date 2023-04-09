""" icon_loader.py

Test gesture icon loader
"""
import handyvision as hv
import cv2 as cv

def main():
    icon_manager: hv.IconManager = hv.IconManager("C:/Users/cnsmith/dev/HandyVision/assets")
    gesture = hv.Gesture.GUN_DOUBLE
    icon_left = icon_manager.icon_for_gesture(hv.Handedness.LEFT, gesture)
    icon_right = icon_manager.icon_for_gesture(hv.Handedness.RIGHT, gesture)

    cv.imshow("GUN DOUBLE LEFT", icon_left)
    cv.imshow("GUN DOUBLE RIGHT", icon_right)

    cv.waitKey(-1)

if __name__ == "__main__":
    main()
