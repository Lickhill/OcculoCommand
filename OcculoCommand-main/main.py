import cv2
import mediapipe as mp
import pyautogui
import time
import os
import warnings
import speech_recognition as sr
import queue
import threading

# ------------------ SETUP ------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=UserWarning)

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0

print("🚀 OCCULO HACKATHON BUILD (DYNAMIC ANCHOR + SMOOTHING)")
print("👁️ Close BOTH eyes for 1.2s to turn ON/OFF")
print("🎙️ Say 'Keyboard on' to toggle typing")
print("⌨️ Commands: 'Delete', 'Enter', 'Space', 'Copy', 'Paste', 'Undo'")
print("🖐️ Keep a palm visible in camera for typing/keyboard commands")
print("-" * 50)

# ------------------ STATE ------------------
cursor_active = False
keyboard_active = False
is_dragging = False
scroll_mode = False

# Click/Drag Timing
last_left_blink_time = 0
left_eye_closed_start = 0
double_click_threshold = 0.4
long_press_threshold = 0.8

# Loop-Proof Toggle State
both_eyes_closed_start = 0
toggle_locked = False      
eyes_open_since = 0        

# Action Flash (For Judges)
action_text = ""
action_time = 0

# --- SENSITIVITY & SMOOTHING ---
screen_w, screen_h = pyautogui.size()
prev_x, prev_y = screen_w // 2, screen_h // 2
anchor_x, anchor_y = 0.5, 0.5  # Will be dynamically set when turned ON

smooth_factor = 0.08  # Lowered heavily (was 0.2). This removes the twitchiness.
SENSITIVITY = 3.5     # Increased. A tiny neck movement covers the screen now.

BLINK_THRESH = 0.008
MOUTH_THRESH = 0.03

# ------------------ SPEECH-TO-TEXT SETUP ------------------
speech_queue = queue.Queue()
recognizer = sr.Recognizer()

def speech_callback(recognizer, audio):
    try:
        text = recognizer.recognize_google(audio)
        print(f"\n🎤 [SPEECH HEARD]: '{text}'")
        print(f"   → Adding to queue for processing")
        speech_queue.put(text)
    except sr.UnknownValueError:
        print("\n🔇 [SPEECH MISSED]: Could not understand audio")
    except sr.RequestError as e:
        print(f"\n⚠️ [SPEECH ERROR]: {e}")

def init_speech_engine():
    try:
        mic = sr.Microphone()
        print("🎤 Calibrating microphone... (2 seconds)")
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=2)
        print("✅ 🎤 Microphone ready! Listening in background...")
        recognizer.listen_in_background(mic, speech_callback)
    except Exception as e:
        print(f"❌ Microphone ERROR: {e}")
        print("   Check if microphone is connected and accessible")

threading.Thread(target=init_speech_engine, daemon=True).start()

# ------------------ INIT ------------------
cam = cv2.VideoCapture(0)
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

def trigger_action(text):
    """Helper to flash text on screen for exactly 1 second"""
    global action_text, action_time
    action_text = text
    action_time = time.time()
    print(f"> {text}")

# ------------------ MAIN LOOP ------------------
while True:
    success, frame = cam.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    frame_h, frame_w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_output = hands.process(rgb_frame)
    palm_visible = bool(hand_output.multi_hand_landmarks)
    output = face_mesh.process(rgb_frame)

    if output.multi_face_landmarks:
        landmarks = output.multi_face_landmarks[0].landmark
        current_time = time.time()

        # -------- 1. EXTRACT LANDMARKS & CALC POSITION --------
        nose = landmarks[1]
        left_eye_top, left_eye_bot = landmarks[159], landmarks[145]
        right_eye_top, right_eye_bot = landmarks[386], landmarks[374]
        
        # --- DYNAMIC ANCHOR MATH ---
        # Instead of 0.5 (dead center of camera), we compare your nose to where it was
        # when you turned the system on (anchor_x, anchor_y).
        rel_x = (nose.x - anchor_x) * SENSITIVITY
        rel_y = (nose.y - anchor_y) * SENSITIVITY
        
        target_x = screen_w * (0.5 + rel_x)
        target_y = screen_h * (0.5 + rel_y)
        
        # Clamp values so PyAutoGUI doesn't crash
        target_x = max(0, min(screen_w, target_x))
        target_y = max(0, min(screen_h, target_y))

        # Heavy Smoothing to fix the "too fast/jerky" issue
        curr_x = prev_x + (target_x - prev_x) * smooth_factor
        curr_y = prev_y + (target_y - prev_y) * smooth_factor

        # -------- 2. MOUTH (SCROLL) --------
        mouth_dist = abs(landmarks[13].y - landmarks[14].y)
        if mouth_dist > MOUTH_THRESH:
            if not scroll_mode: 
                trigger_action("🟡 SCROLLING")
            scroll_mode = True
            pyautogui.scroll(int((curr_y - prev_y) * -5))
        else:
            scroll_mode = False
            if cursor_active:
                pyautogui.moveTo(curr_x, curr_y)
        prev_x, prev_y = curr_x, curr_y

        # -------- 3. EYE DISTANCES --------
        left_dist = abs(left_eye_top.y - left_eye_bot.y)
        right_dist = abs(right_eye_top.y - right_eye_bot.y)
        both_closed = left_dist < BLINK_THRESH and right_dist < BLINK_THRESH

        # -------- 4. THE PROGRESS BAR TOGGLE --------
        if both_closed:
            if not toggle_locked:
                if both_eyes_closed_start == 0:
                    both_eyes_closed_start = current_time
                
                elapsed = current_time - both_eyes_closed_start
                
                # DRAW PROGRESS BAR
                progress = min(1.0, elapsed / 1.2)
                bar_w = 300
                cv2.rectangle(frame, (frame_w//2 - bar_w//2, frame_h//2 + 20), (frame_w//2 + bar_w//2, frame_h//2 + 40), (50, 50, 50), -1)
                cv2.rectangle(frame, (frame_w//2 - bar_w//2, frame_h//2 + 20), (frame_w//2 - bar_w//2 + int(bar_w * progress), frame_h//2 + 40), (0, 165, 255), -1)
                
                remaining = max(0, 1.2 - elapsed)
                cv2.putText(frame, f"TOGGLING SYSTEM: {remaining:.1f}s", (frame_w//2 - 120, frame_h//2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

                if elapsed >= 1.2:
                    cursor_active = not cursor_active 
                    trigger_action(f"🟢 SYSTEM ON" if cursor_active else "🔴 SYSTEM OFF")
                    toggle_locked = True
                    both_eyes_closed_start = 0 
                    
                    # !! DYNAMIC CALIBRATION TRIGGER !!
                    # When turned ON, set current nose pos as the center of the screen
                    if cursor_active:
                        anchor_x, anchor_y = nose.x, nose.y
                        prev_x, prev_y = screen_w // 2, screen_h // 2
            else:
                cv2.putText(frame, "RELEASE TO RESET", (frame_w//2 - 100, frame_h//2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            eyes_open_since = 0 
        else:
            if eyes_open_since == 0:
                eyes_open_since = current_time
            
            if current_time - eyes_open_since > 0.2:
                toggle_locked = False
                both_eyes_closed_start = 0

        # -------- 5. MOUSE ACTIONS --------
        if cursor_active:
            if left_dist < BLINK_THRESH and not both_closed:
                if left_eye_closed_start == 0:
                    left_eye_closed_start = current_time

                if not is_dragging and (current_time - left_eye_closed_start) > long_press_threshold:
                    pyautogui.mouseDown()
                    is_dragging = True
                    trigger_action("📂 DRAG START")

            elif left_eye_closed_start != 0:
                if is_dragging:
                    pyautogui.mouseUp()
                    is_dragging = False
                    trigger_action("📂 DRAG DROP")
                else:
                    if (current_time - last_left_blink_time) < double_click_threshold:
                        pyautogui.doubleClick()
                        trigger_action("🖱️ DOUBLE CLICK")
                    else:
                        pyautogui.click()
                        trigger_action("🖱️ LEFT CLICK")
                    last_left_blink_time = current_time
                left_eye_closed_start = 0

            if right_dist < BLINK_THRESH and not both_closed:
                pyautogui.rightClick()
                trigger_action("🖱️ RIGHT CLICK")
                time.sleep(0.4)

        # -------- 6. CLEAN ACTION FLASH --------
        if current_time - action_time < 1.2:
            cv2.putText(frame, action_text, (frame_w//2 - 120, frame_h - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

    # -------- 7. UI OVERLAY --------
    cv2.putText(frame, f"CURSOR: {'ON' if cursor_active else 'OFF'}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if cursor_active else (0, 0, 255), 2)
    cv2.putText(frame, f"KEYBOARD: {'ON' if keyboard_active else 'OFF'}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if keyboard_active else (0, 0, 255), 2)
    cv2.putText(frame, f"PALM: {'VISIBLE' if palm_visible else 'NOT VISIBLE'}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if palm_visible else (0, 0, 255), 2)

    # -------- 8. SPEECH COMMANDS & KEYBOARD --------
    while not speech_queue.empty():
        raw_text = speech_queue.get()
        command_text = raw_text.strip().lower().replace(".", "").replace("!", "").replace("?", "")
        
        print(f"\n📋 [COMMAND PROCESSING]")
        print(f"   Raw: '{raw_text}'")
        print(f"   Cleaned: '{command_text}'")
        print(f"   Keyboard Status: {'ON' if keyboard_active else 'OFF'}")
        print(f"   Palm Visible: {'YES' if palm_visible else 'NO'}")

        if command_text in ["keyboard on", "turn on keyboard", "keyboard mode"]:
            keyboard_active = True
            trigger_action("✅ 🎙️ KEYBOARD ON")
            print(f"   ✅ Keyboard toggled to: ON")
            continue
        elif command_text in ["keyboard off", "turn off keyboard", "keyboard mode off"]:
            keyboard_active = False
            trigger_action("❌ 🔇 KEYBOARD OFF")
            print(f"   ❌ Keyboard toggled to: OFF")
            continue

        if not keyboard_active:
            print(f"   ⏭️ [IGNORED]: Keyboard is OFF - use 'keyboard on' to enable")
            continue

        shortcut_commands = {
            "delete",
            "backspace",
            "enter",
            "new line",
            "space",
            "spacebar",
            "select all",
            "copy",
            "paste",
            "undo",
        }

        if command_text not in shortcut_commands:
            if palm_visible:
                print("   ⏭️ [IGNORED]: Hide the palm to type normal dictation")
                continue

            pyautogui.write(raw_text, interval=0.01)
            trigger_action(f"⌨️ {raw_text}")
            print(f"   ✅ Typed text: '{raw_text}'")
            continue

        if not palm_visible:
            print("   ⏭️ [IGNORED]: Show a palm to use action buttons")
            continue

        # --- KEYBOARD COMMANDS ---
        if command_text == "delete":
            pyautogui.hotkey('ctrl', 'backspace')
            trigger_action("⌨️ [DELETE WORD]")
            print(f"   ✅ DELETE WORD executed")
        elif command_text == "backspace":
            pyautogui.press('backspace')
            trigger_action("⌨️ [BACKSPACE]")
            print(f"   ✅ BACKSPACE executed")
        elif command_text in ["enter", "new line"]:
            pyautogui.press('enter')
            trigger_action("⌨️ [ENTER]")
            print(f"   ✅ ENTER executed")
        elif command_text in ["space", "spacebar"]:
            pyautogui.press('space')
            trigger_action("⌨️ [SPACE]")
            print(f"   ✅ SPACE executed")
        elif command_text == "select all":
            pyautogui.hotkey('ctrl', 'a')
            trigger_action("⌨️ [SELECT ALL]")
            print(f"   ✅ SELECT ALL executed")
        elif command_text == "copy":
            pyautogui.hotkey('ctrl', 'c')
            trigger_action("⌨️ [COPY]")
            print(f"   ✅ COPY executed")
        elif command_text == "paste":
            pyautogui.hotkey('ctrl', 'v')
            trigger_action("⌨️ [PASTE]")
            print(f"   ✅ PASTE executed")
        elif command_text == "undo":
            pyautogui.hotkey('ctrl', 'z')
            trigger_action("⌨️ [UNDO]")
            print(f"   ✅ UNDO executed")

    cv2.imshow("OCCULO", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()