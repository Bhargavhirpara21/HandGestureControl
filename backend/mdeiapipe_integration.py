# import cv2
# import mediapipe as mp
# import requests
# import numpy as np
#
# # Initialize MediaPipe Hands and Drawing Utilities
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
# mp_draw = mp.solutions.drawing_utils
#
# # Initialize webcam
# cap = cv2.VideoCapture(0)
# api_url = "http://127.0.0.1:5000/predict"
#
# print("Press 'q' to quit the application")
#
# while True:
#     # Capture frame from webcam
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to capture frame. Exiting...")
#         break
#
#     # Convert the frame to RGB (MediaPipe requires RGB format)
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#     # Process the frame to detect hand landmarks
#     results = hands.process(frame_rgb)
#
#     # If hands are detected, process landmarks
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             # Draw landmarks on the frame
#             mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#
#             # Extract landmark data
#             vector = []
#             for lm in hand_landmarks.landmark:
#                 vector.extend([lm.x, lm.y, lm.z])
#
#             # Send the vector to the API
#             try:
#                 response = requests.post(api_url, json={"vector": vector})
#                 if response.ok:
#                     gesture = response.json().get("gesture")
#                     print(f"Detected Gesture: {gesture}")
#                 else:
#                     print(f"API Error: {response.status_code}")
#             except Exception as e:
#                 print(f"Error communicating with API: {str(e)}")
#
#     # Display the frame with annotations
#     cv2.imshow("Hand Tracking", frame)
#
#     # Break the loop when 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Release the webcam and close OpenCV windows
# cap.release()
# cv2.destroyAllWindows()



#working
import cv2
import mediapipe as mp
import requests
import numpy as np

# Initialize MediaPipe Hands and Drawing Utilities
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)
api_url = "http://127.0.0.1:5000/predict"

print("Streaming... Press 'q' to quit.")

gesture_text = "No Gesture Detected"

# Set up full-screen window
cv2.namedWindow("Hand Gesture Detection", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Hand Gesture Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Convert the frame to RGB (MediaPipe requires RGB format)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hand landmarks
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmark data
            vector = []
            for lm in hand_landmarks.landmark:
                vector.extend([lm.x, lm.y, lm.z])

            # Send the vector to the API
            try:
                response = requests.post(api_url, json={"vector": vector})
                if response.ok:
                    gesture_text = response.json().get("gesture", "No Gesture Detected")
                    print(f"Detected Gesture: {gesture_text}")  # Log the gesture
                else:
                    gesture_text = "API Error"
                    print("API Error")  # Log the error
            except Exception as e:
                gesture_text = "Error Communicating with API"
                print(f"Error Communicating with API: {e}")  # Log the exception

    # Display the gesture text on the frame
    cv2.putText(frame, f"Gesture: {gesture_text}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the frame in full screen
    cv2.imshow("Hand Gesture Detection", frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()







