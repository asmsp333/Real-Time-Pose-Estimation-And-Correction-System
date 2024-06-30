import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize Mediapipe and OpenCV Components
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

# Initialize the pose model and hand model
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# Function to display text on the image
def display_text(image, text, position, size=1, color=(0, 0, 255)):
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, size, color, 2, cv2.LINE_AA)

# Initialize counters for exercise repetitions and stage
pushup_counter = 0
plank_counter = 0
squat_counter = 0
stage = None
ask_permission = False
exercise_mode = "pushups"

# Hand sign counters
pushups_sign_counter = 0
planks_sign_counter = 0
squats_sign_counter = 0
SIGN_HOLD_FRAMES = 30

# Capture Video Stream
cap = cv2.VideoCapture(0)

# Countdown before starting exercise selection
start_time = time.time()
while time.time() - start_time < 5:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Flip the frame horizontally for a mirror view
    countdown = 5 - int(time.time() - start_time)
    display_text(frame, f'Starting Exercise Selection in {countdown}', (50, 50))
    cv2.imshow('Exercise Selection', frame)
    cv2.waitKey(100)

# Timer for exercise
exercise_start_time = time.time()

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a mirror view
        frame = cv2.flip(frame, 1)

        # Convert the image from BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Process the image to detect the pose and hands
        results_pose = pose.process(image)
        results_hands = hands.process(image)

        # Draw the pose annotations on the image
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Hand sign detection for exercise selection
        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                
                # Check for hand signs: thumb up for push-ups, index finger up for planks, thumb and index up for squats
                if thumb_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y and index_tip.y > hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y:
                    pushups_sign_counter += 1
                    planks_sign_counter = 0
                    squats_sign_counter = 0
                    if pushups_sign_counter >= SIGN_HOLD_FRAMES:
                        exercise_mode = "pushups"
                        display_text(image, "Push-ups Mode Selected", (50, 50), size=1, color=(0, 255, 0))
                elif index_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y and thumb_tip.y > hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y:
                    planks_sign_counter += 1
                    pushups_sign_counter = 0
                    squats_sign_counter = 0
                    if planks_sign_counter >= SIGN_HOLD_FRAMES:
                        exercise_mode = "planks"
                        display_text(image, "Planks Mode Selected", (50, 50), size=1, color=(0, 255, 0))
                elif thumb_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y and index_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y:
                    squats_sign_counter += 1
                    pushups_sign_counter = 0
                    planks_sign_counter = 0
                    if squats_sign_counter >= SIGN_HOLD_FRAMES:
                        exercise_mode = "squats"
                        display_text(image, "Squats Mode Selected", (50, 50), size=1, color=(0, 255, 0))
                else:
                    pushups_sign_counter = 0
                    planks_sign_counter = 0
                    squats_sign_counter = 0

        try:
            landmarks = results_pose.pose_landmarks.landmark
            
            # Handle push-up repetitions
            if exercise_mode == "pushups" and not ask_permission:
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                angle = calculate_angle(shoulder, elbow, wrist)
                
                # Real-time feedback for push-up repetitions
                if angle > 160:
                    stage = "down"
                if angle < 90 and stage == 'down':
                    stage = "up"
                    pushup_counter += 1
                    print(f"Push-up {pushup_counter}: Correct")

                # Display angle and feedback on the image for push-ups
                cv2.putText(image, str(angle), 
                            tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                cv2.putText(image, 'STAGE', (65, 12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, stage, 
                            (65, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

                cv2.putText(image, 'PUSH-UPS', (10, 12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(pushup_counter), 
                            (10, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

                # Check if push-up exercise is completed
                if pushup_counter >= 5:
                    display_text(image, 'Do you want to repeat? (y/n)', (50, 100))
                    cv2.imshow('Pose Estimation', image)
                    ask_permission = True
            
            # Handle plank hold duration
            elif exercise_mode == "planks" and not ask_permission:
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                angle = calculate_angle(shoulder, hip, ankle)
                
                # Real-time feedback for plank hold duration
                if 150 < angle < 180:
                    plank_counter += 1
                    print(f"Plank duration: {plank_counter} frames")

                # Display angle and feedback on the image for planks
                cv2.putText(image, str(angle), 
                            tuple(np.multiply(hip, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                cv2.putText(image, 'PLANKS', (10, 12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(plank_counter // 10), 
                            (10, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

                # Check if plank hold is completed
                if plank_counter >= 300:  # Assuming 30 seconds at 10 FPS
                    display_text(image, 'Do you want to repeat? (y/n)', (50, 100))
                    cv2.imshow('Pose Estimation', image)
                    ask_permission = True

            # Handle squat repetitions
            elif exercise_mode == "squats" and not ask_permission:
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                angle = calculate_angle(hip, knee, ankle)
                
                # Real-time feedback for squat repetitions
                if angle > 160:
                    stage = "up"
                if angle < 90 and stage == 'up':
                    stage = "down"
                    squat_counter += 1
                    print(f"Squat {squat_counter}: Correct")

                # Display angle and feedback on the image for squats
                cv2.putText(image, str(angle), 
                            tuple(np.multiply(knee, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                cv2.putText(image, 'STAGE', (65, 12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, stage, 
                            (65, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

                cv2.putText(image, 'SQUATS', (10, 12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(squat_counter), 
                            (10, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

                # Check if squat exercise is completed
                if squat_counter >= 10:
                    display_text(image, 'Do you want to repeat? (y/n)', (50, 100))
                    cv2.imshow('Pose Estimation', image)
                    ask_permission = True

            # Ask for permission to repeat after the exercise is completed
            if ask_permission:
                key = cv2.waitKey(0) & 0xFF
                if key == ord('y'):
                    pushup_counter = 0
                    plank_counter = 0
                    squat_counter = 0
                    stage = None
                    ask_permission = False
                    exercise_start_time = time.time()  # Reset timer for new set
                elif key == ord('n'):
                    break

        except Exception as e:
            print(f"An error occurred: {e}")
            pass

        mp_drawing.draw_landmarks(image, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imshow('Pose Estimation', image)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break

finally:
    # Calculate total exercise time
    exercise_end_time = time.time()
    total_time = exercise_end_time - exercise_start_time

    # Display motivational message
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Flip the frame horizontally for a mirror view
    display_text(frame, f"Great job! You completed your exercise in {total_time:.2f} seconds.", (50, 50), size=1, color=(0, 255, 0))
    display_text(frame, "Keep pushing your limits!", (50, 100), size=1, color=(0, 255, 0))
    cv2.imshow('Pose Estimation', frame)
    cv2.waitKey(5000)  # Display the message for 5 seconds

    cap.release()
    cv2.destroyAllWindows()
    print(f"Great job! You completed your exercise in {total_time:.2f} seconds. Keep pushing your limits!")
