import cv2
import mediapipe as mp
import pyautogui
import time
import os
import warnings

# ------------------ SETUP ------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=UserWarning)

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0

print("🚀 OCCULO HACKATHON BUILD (DYNAMIC ANCHOR + SMOOTHING)")
print("👁️ Close BOTH eyes for 1.2s to turn ON/OFF")

# ------------------ STATE ------------------
cursor_active = False
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

# ------------------ INIT ------------------
cam = cv2.VideoCapture(0)
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True)

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

    cv2.imshow("OCCULO", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()