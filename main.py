import mss
import numpy as np
import cv2

def screen_record():
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        width = monitor["width"]
        height = monitor["height"]

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('screen_capture.avi', fourcc, 20.0, (width, height))

        while True:
            img = np.array(sct.grab(monitor))
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            out.write(img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release resources
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    screen_record()