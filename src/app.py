import cv2

def main():
    vid = cv2.VideoCapture(1)
    while (True):
        ret, frame = vid.read()
        cv2.imshow("Webcam Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vid.release()
    cv2.destoryAllWindows()

if __name__=="__main__":
    main()
