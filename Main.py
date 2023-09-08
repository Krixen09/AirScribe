import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Open the camera
cap = cv2.VideoCapture(0)

# Create a blank canvas to draw on with a transparent background
canvas = np.zeros((480, 640, 4), dtype=np.uint8)

# Set the alpha channel to 255 (fully opaque)
canvas[:, :, 3] = 255

# Initialize variables for drawing and erasing
drawing = False
erasing = False
prev_point = None
line_color = (0, 0, 0, 255)  # Default line color is black with full opacity

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(frame_rgb)

    # Display the canvas for drawing
    frame = cv2.addWeighted(frame, 1, canvas[:, :, :3], 0.5, 0)

    # Detect the index finger landmarks
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            for i, landmark in enumerate(landmarks.landmark):
                if i == 8:  # Index finger tip landmark
                    x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])

                    # Erase on the canvas if erasing mode is enabled
                    if erasing:
                        cv2.circle(canvas, (x, y), 10, (0, 0, 0, 0), -1)  # Set pixel to transparent
                    else:
                        # Draw on the canvas if drawing mode is enabled
                        if drawing:
                            cv2.circle(canvas, (x, y), 5, line_color, -1)
                            if prev_point is not None:
                                cv2.line(canvas, prev_point, (x, y), line_color, 5)
                    prev_point = (x, y)

    # Display the frame with drawing
    cv2.imshow("AirScribe", frame)

    # User Interaction
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = np.zeros_like(canvas)  # Clear the canvas by setting alpha to 0 (fully transparent)
    elif key == ord('d'):
        drawing = not drawing
        erasing = False  # Turn off erasing mode when switching to drawing
    elif key == ord('e'):
        erasing = not erasing
        drawing = False  # Turn off drawing mode when switching to erasing
    elif key == ord('1'):
        line_color = (0, 0, 255, 255)  # Red with full opacity
    elif key == ord('2'):
        line_color = (0, 255, 0, 255)  # Green with full opacity
    elif key == ord('3'):
        line_color = (255, 0, 0, 255)  # Blue with full opacity

# Release the camera and close the OpenCV window
cap.release()
cv2.destroyAllWindows()

# Close the MediaPipe Hands module
hands.close()
