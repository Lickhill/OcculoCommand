import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
import pyautogui
import speech_recognition as sr
import threading
import time
from pathlib import Path
from urllib.request import urlretrieve


recognizer = sr.Recognizer()
cursor_active = False

prev_x, prev_y = 0, 0
smooth_factor = 0.2

FACE_LANDMARKER_MODEL_URL = "https://storage.googleapis.com/mediapipe-assets/face_landmarker.task"
FACE_LANDMARKER_MODEL_PATH = Path(__file__).with_name("face_landmarker.task")


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
                except Exception as exc:
                    print("Speech error:", exc)

    except (AttributeError, OSError) as exc:
        print("Microphone unavailable:", exc)


def load_face_landmarker():
    if not FACE_LANDMARKER_MODEL_PATH.exists():
        print("Downloading face landmark model...")
        urlretrieve(FACE_LANDMARKER_MODEL_URL, FACE_LANDMARKER_MODEL_PATH)

    options = vision.FaceLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(FACE_LANDMARKER_MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(options)


threading.Thread(target=listen_for_commands, daemon=True).start()

cam = cv2.VideoCapture(0)
try:
    face_landmarker = load_face_landmarker()
except Exception as exc:
    print("Face tracking unavailable:", exc)
    raise SystemExit(1)

screen_w, screen_h = pyautogui.size()

while True:
    success, frame = cam.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    output = face_landmarker.detect(mp_image)

    if output.face_landmarks:
        landmarks = output.face_landmarks[0]
        frame_h, frame_w, _ = frame.shape

        if cursor_active:
            nose = landmarks[1]

            x = int(nose.x * frame_w)
            y = int(nose.y * frame_h)

            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            screen_x = nose.x * screen_w
            screen_y = nose.y * screen_h

            curr_x = prev_x + (screen_x - prev_x) * smooth_factor
            curr_y = prev_y + (screen_y - prev_y) * smooth_factor

            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y

        left_top = landmarks[159]
        left_bottom = landmarks[145]

        x1 = int(left_top.x * frame_w)
        y1 = int(left_top.y * frame_h)
        x2 = int(left_bottom.x * frame_w)
        y2 = int(left_bottom.y * frame_h)

        cv2.circle(frame, (x1, y1), 3, (0, 255, 255), -1)
        cv2.circle(frame, (x2, y2), 3, (0, 255, 255), -1)

        blink_distance = abs(left_top.y - left_bottom.y)

        if cursor_active and blink_distance < 0.008:
            pyautogui.click()
            print("CLICK")
            time.sleep(0.5)

    cv2.imshow("Eye Controlled Mouse", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()