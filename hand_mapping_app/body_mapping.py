import cv2
import mediapipe as mp

# Inicjalizacja modeli Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

# Otwieranie kamery (lub wideo)
cap = cv2.VideoCapture(0)  # 0 to numer kamery, jeśli używasz kamerki internetowej

# Ustawienie rozdzielczości na 1920x1080
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Ustawienie modeli Pose i Hands
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
     mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Przetwarzanie obrazu w celu wykrycia pozycji ciała i dłoni
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Wykrycie pozycji ciała
        results_pose = pose.process(image)
        # Wykrycie dłoni
        results_hands = hands.process(image)
        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Rysowanie szkieletu ciała, jeśli wykryto
        if results_pose.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Rysowanie szkieletu dłoni, jeśli wykryto
        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Wyświetlanie obrazu
        cv2.imshow('Szkielet ciała i dłoni', image)

        # Zakończenie działania po naciśnięciu 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
