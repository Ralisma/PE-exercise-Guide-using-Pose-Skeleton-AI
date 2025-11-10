import cv2
import numpy as np
from config import WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT

class DisplayManager:
    def __init__(self):
        self.close_flag = False
        cv2.startWindowThread()
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)
        cv2.moveWindow(WINDOW_NAME, 100, 100)
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
        # --- FIX 1: "Fails to Open" ---
        # Increased wait time from 1ms to 50ms.
        # This gives the OS more time to process the window creation
        # events before the main loop starts.
        cv2.waitKey(50)
        print("Display window created and ready.")

    def show_frame(self, frame, wait_ms=1):
        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(wait_ms) & 0xFF
        if key == ord('x'):
            self.close_flag = True

    def should_close(self):
        return self.close_flag

    def show_completion_frame(self):
        display_frame = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
        cv2.putText(display_frame, "WORKOUT COMPLETE!", (200, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(display_frame, "All exercises completed!", (220, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display_frame, "Rest and good work!", (280, 350),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display_frame, "Press 'x' to exit", (320, 400),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        while not self.close_flag:
            # --- FIX 2: "Fails to Quit" ---
            # Added a 100ms wait. The original code called show_frame
            # with the default 1ms, creating a busy-wait loop
            # that made the 'x' key unresponsive.
            self.show_frame(display_frame, wait_ms=100)
