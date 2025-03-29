import cv2
import mediapipe as mp
import pyautogui
import speech_recognition as sr
import threading
import time

# Initialize speech recognizer
recognizer = sr.Recognizer()

# Global flag to control cursor movement
cursor_active = False


def listen_for_commands():
    global cursor_active
    while True:
        with sr.Microphone() as source:
            print("Listening for commands...")
            audio = recognizer.listen(source)
            try:
                command = recognizer.recognize_google(audio).lower()
                words = command.split()
                print(f"Recognized: {command}")
                if any(word in {"off", "of"} for word in words):
                    cursor_active = False
                    print("Cursor movement deactivated")
                # Then check for activation
                elif "on" in words:
                    cursor_active = True
                    print("Cursor movement activated")
            except sr.UnknownValueError:
                pass
            except sr.RequestError:
                print("Could not request results from speech recognition service")


# Start the speech recognition thread
threading.Thread(target=listen_for_commands, daemon=True).start()

cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()
sensitivity = 2.0  # Reduced sensitivity
pyautogui.PAUSE = 0.1  # Add a small delay between PyAutoGUI actions

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape

    if landmark_points and cursor_active:
        landmarks = landmark_points[0].landmark
        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0))

            if id == 1:
                adj_x = (landmark.x - 0.5) * sensitivity + 0.5
                adj_y = (landmark.y - 0.5) * sensitivity + 0.5

                screen_x = max(10, min(screen_w - 10, screen_w * adj_x))
                screen_y = max(10, min(screen_h - 10, screen_h * adj_y))

                # Implement boundary checking
                if 10 < screen_x < screen_w - 10 and 10 < screen_y < screen_h - 10:
                    pyautogui.moveTo(screen_x, screen_y)

        left = [landmarks[145], landmarks[159]]
        for landmark in left:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255))

        if (left[0].y - left[1].y) < 0.009:
            pyautogui.click()
            time.sleep(1)  # Use time.sleep instead of pyautogui.sleep

    cv2.imshow("Eye Controlled Mouse", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
