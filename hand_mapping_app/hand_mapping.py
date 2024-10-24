import cv2
import mediapipe as mp
import time
import pyautogui

# Inicjalizacja mediapipe
mp_pose = mp.solutions.pose.Pose()
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands.Hands()

# Otwarcie kamery
cap = cv2.VideoCapture(0)

# Ustawienie rozdzielczości 16:9
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Zmienna do przechowywania ostatniej pozycji palca wskazującego
last_index_finger_tip_x = 0
last_index_finger_tip_y = 0

# Pętla główna
while True:
    # Odczytanie klatki z kamery
    success, image = cap.read()
    if not success:
        break

    # Konwersja obrazu do RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Przetwarzanie obrazu przez model mediapipe dla pozycji ciała
    results_pose = mp_pose.process(image)

    # Przetwarzanie obrazu przez model mediapipe dla dłoni
    results_hands = mp_hands.process(image)

    # Konwersja obrazu z powrotem do BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Rysowanie punktów i linii na obrazie dla pozycji ciała
    if results_pose.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results_pose.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

    # Rysowanie punktów i linii na obrazie dla dłoni
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

            # Wyświetlanie wartości dla każdego punktu
            for i, landmark in enumerate(hand_landmarks.landmark):
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                cv2.putText(image, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Pobranie współrzędnych palca wskazującego
            index_finger_tip_x = int(hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].x * image.shape[1])
            index_finger_tip_y = int(hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].y * image.shape[0])

            # Obliczenie różnicy między obecną a poprzednią pozycją
            x_diff = index_finger_tip_x - last_index_finger_tip_x
            y_diff = index_finger_tip_y - last_index_finger_tip_y

            # Sprawdzenie, czy ręka jest po lewej stronie
            if index_finger_tip_x < image.shape[1] / 2:
                # Odwrócenie osi x i y
                pyautogui.moveRel(-y_diff, -x_diff, duration=0.01)
            else:
                # Odwrócenie osi x i y
                pyautogui.moveRel(y_diff, x_diff, duration=0.01)

            # Zapamiętanie ostatniej pozycji palca wskazującego
            last_index_finger_tip_x = index_finger_tip_x
            last_index_finger_tip_y = index_finger_tip_y

    # Wyświetlenie obrazu
    cv2.imshow('Ciało', image)

    # Zakończenie pętli, jeśli naciśnięto klawisz 'q'
    if cv2.waitKey(1) == ord('q'):
        break

# Zamknięcie kamery i okna
cap.release()
cv2.destroyAllWindows()
