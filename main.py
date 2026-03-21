import cv2
import mediapipe as mp
import pyautogui
import speech_recognition as sr
import threading
import time

# Initialize
recognizer = sr.Recognizer()
cursor_active = False

# Smooth movement
prev_x, prev_y = 0, 0
smooth_factor = 0.2


# 🎤 Voice control function (FIXED)
def listen_for_commands():
    global cursor_active

    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)

            while True:
                print("Listening...")
                try:
                    audio = recognizer.listen(source, timeout=5)
                    command = recognizer.recognize_google(audio).lower()
                    print("Command:", command)

                    if "off" in command:
                        cursor_active = False
                        print("Cursor OFF")

                    elif "on" in command:
                        cursor_active = True
                        print("Cursor ON")

                except sr.WaitTimeoutError:
                    continue
                except sr.UnknownValueError:
                    print("Didn't catch that")
                except Exception as e:
                    print("Speech error:", e)

    except Exception as e:
        print("Microphone error:", e)


# Start voice thread
threading.Thread(target=listen_for_commands, daemon=True).start()

# Camera + FaceMesh
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

screen_w, screen_h = pyautogui.size()

while True:
    success, frame = cam.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    output = face_mesh.process(rgb_frame)

    if output.multi_face_landmarks:
        landmarks = output.multi_face_landmarks[0].landmark
        frame_h, frame_w, _ = frame.shape

        # 🔹 Cursor movement (NOSE - stable)
        if cursor_active:
            nose = landmarks[1]

            x = int(nose.x * frame_w)
            y = int(nose.y * frame_h)

            cv2.circle(frame, (x, y), 5, (0, 255, 0))

            screen_x = nose.x * screen_w
            screen_y = nose.y * screen_h

            # Smooth movement
            curr_x = prev_x + (screen_x - prev_x) * smooth_factor
            curr_y = prev_y + (screen_y - prev_y) * smooth_factor

            pyautogui.moveTo(curr_x, curr_y)

            prev_x, prev_y = curr_x, curr_y

        # 🔹 Blink detection (LEFT EYE)
        left_top = landmarks[159]
        left_bottom = landmarks[145]

        x1 = int(left_top.x * frame_w)
        y1 = int(left_top.y * frame_h)
        x2 = int(left_bottom.x * frame_w)
        y2 = int(left_bottom.y * frame_h)

        cv2.circle(frame, (x1, y1), 3, (0, 255, 255))
        cv2.circle(frame, (x2, y2), 3, (0, 255, 255))

        blink_distance = abs(left_top.y - left_bottom.y)

        if cursor_active and blink_distance < 0.008:
            pyautogui.click()
            print("CLICK")
            time.sleep(0.5)

    # Display
    cv2.imshow("Eye Controlled Mouse", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
