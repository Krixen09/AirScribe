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

# Define a list of colors
colors = [(0, 0, 255, 255), (0, 255, 0, 255), (255, 0, 0, 255), (0, 255, 255, 255), (255, 0, 255, 255), (255, 255, 0, 255)]

# Define the eraser color (fully transparent)
eraser_color = (0, 0, 0, 0)

# Calculate the dimensions of each color swatch and the spacing
swatch_width = 40
swatch_height = 40
swatch_spacing = 10

# Initialize variables for drawing and erasing
drawing = False
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

    # Display the color palette on the left side of the frame
    swatch_x = swatch_spacing
    swatch_y = swatch_spacing
    
    for color in colors:
        cv2.circle(frame, (swatch_x + swatch_width // 2, swatch_y + swatch_height // 2), swatch_width // 2, color, -1)
        swatch_y += swatch_height + swatch_spacing

    # Display the eraser tool
    cv2.circle(frame, (swatch_x + swatch_width // 2, swatch_y + swatch_height // 2), swatch_width // 2, eraser_color, -1)

    # Detect the index finger landmarks
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            for i, landmark in enumerate(landmarks.landmark):
                if i == 8:  # Index finger tip landmark
                    x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])

                    # Check if the user touched the eraser tool
                    if swatch_x <= x <= swatch_x + swatch_width and swatch_y <= y <= swatch_y + swatch_height:
                        line_color = eraser_color  # Activate eraser
                    else:
                        # Check if the user touched a color swatch
                        for idx, color in enumerate(colors):
                            swatch_xi = swatch_x
                            swatch_yi = swatch_spacing + idx * (swatch_height + swatch_spacing)
                            if swatch_xi <= x <= swatch_xi + swatch_width and swatch_yi <= y <= swatch_yi + swatch_height:
                                line_color = colors[idx]  # Change the drawing color
                                break

                    # Start drawing
                    if drawing:
                        cv2.circle(canvas, (x, y), 5, line_color, -1)
                        if prev_point is not None:
                            cv2.line(canvas, prev_point, (x, y), line_color, 5)
                    prev_point = (x, y)

    # Display the frame with drawing and the color palette (flipped horizontally)
    cv2.imshow("AirScribe", cv2.flip(frame, 1))
 

    # User Interaction
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('d'):
        drawing = not drawing

    elif key == ord('c'):
        canvas = np.zeros_like(canvas) # Clear the canvas by setting alpha to 0 (fully transparent)

# Release the camera and close the OpenCV window
cap.release()
cv2.destroyAllWindows()

# Close the MediaPipe Hands module
hands.close()
