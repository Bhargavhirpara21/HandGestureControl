'''
import cv2
import mediapipe as mp
import csv
import pandas as pd

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)

data = []
label = int(input("Enter gesture label (0 for two fingers, 1 for one finger): "))

print("Collecting data... Press 'q' to stop.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            vector = []
            for lm in hand_landmarks.landmark:
                vector.extend([lm.x, lm.y, lm.z])
            data.append(vector + [label])

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save data to a CSV file
with open('gesture_data.csv', 'a', newline='') as f:  # Append to existing file
    writer = csv.writer(f)
    writer.writerows(data)

print(f"Data for label {label} saved.")


data = pd.read_csv('gesture_data.csv')
print(data.head())
print("\nLabel Distribution:")
print(data.iloc[:, -1].value_counts())

# Extract the first feature vector (row 0, without the label column)
vector = data.iloc[0, :-1].tolist()
print("Feature Vector:", vector)
'''



# import cv2
# import mediapipe as mp
# import csv
# import pandas as pd
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
#
# cap = cv2.VideoCapture(0)
#
# data = []
# label = int(input("Enter gesture label (0 for two fingers, 1 for one finger): "))
#
# print("Collecting data... Press 'q' to stop.")
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = hands.process(frame_rgb)
#
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             vector = []
#             for lm in hand_landmarks.landmark:
#                 vector.extend([lm.x, lm.y, lm.z])
#             data.append(vector + [label])
#
#     cv2.imshow("Webcam", frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()
#
# # Save data to a CSV file
# with open('gesture_data.csv', 'a', newline='') as f:  # Append to existing file
#     writer = csv.writer(f)
#     writer.writerows(data)
#
# print(f"Data for label {label} saved.")
#
# data = pd.read_csv('gesture_data.csv')
# print(data.head())
# print("\nLabel Distribution:")
# print(data.iloc[:, -1].value_counts())


#working
import cv2
import mediapipe as mp
import csv
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)

data = []
label = int(input("Enter gesture label (0 for other gestures, 1 for one finger, 2 for two fingers): "))
num_samples = 20  # Collect 20 samples
interval = 1     # Wait for 1 seconds between samples

print(f"Collecting {num_samples} samples for label {label} every {interval} seconds. Press 'q' to quit.")

samples_collected = 0

while samples_collected < num_samples:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            vector = []
            for lm in hand_landmarks.landmark:
                vector.extend([lm.x, lm.y, lm.z])

            data.append(vector + [label])
            samples_collected += 1
            print(f"Collected sample {samples_collected}/{num_samples}")

            # Wait for the interval before collecting the next sample
            time.sleep(interval)

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save data to CSV
with open('gesture_data.csv', 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(data)

print(f"Data for label {label} saved.")








