import cv2
import mediapipe as mp
import pyautogui

# Inicjalizacja mediapipe
mp_hands = mp.solutions.hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Otwarcie kamery
cap = cv2.VideoCapture(0)

# Ustawienie rozdzielczości kamery (np. 1280x720)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Pobranie rozdzielczości ekranu
screen_width, screen_height = pyautogui.size()

# Zmienna do śledzenia pozycji kursora
last_cursor_x, last_cursor_y = pyautogui.position()

# Parametr wygładzania ruchu
smooth_factor = 0.2

# Zmienne do śledzenia stanu uszczypnięcia
is_left_pinching = False
is_right_pinching = False

try:
    # Pętla główna
    while True:
        # Odczytanie klatki z kamery
        success, image = cap.read()
        if not success:
            break

        # Konwersja obrazu do RGB i przetwarzanie przez model Mediapipe dla dłoni
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results_hands = mp_hands.process(image_rgb)

        # Jeśli wykryto dłonie, przetwarzaj współrzędne
        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

                # Pobranie współrzędnych palca wskazującego, środkowego i kciuka
                cam_width, cam_height = image.shape[1], image.shape[0]
                index_finger_tip_x = int(hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].x * cam_width)
                index_finger_tip_y = int(hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].y * cam_height)
                middle_finger_tip_x = int(hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP].x * cam_width)
                middle_finger_tip_y = int(hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP].y * cam_height)
                thumb_tip_x = int(hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP].x * cam_width)
                thumb_tip_y = int(hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP].y * cam_height)

                # Obliczenie odległości między palcem wskazującym a kciukiem
                left_distance = ((index_finger_tip_x - thumb_tip_x) ** 2 + (index_finger_tip_y - thumb_tip_y) ** 2) ** 0.5

                # Obliczenie odległości między palcem środkowym a kciukiem
                right_distance = ((middle_finger_tip_x - thumb_tip_x) ** 2 + (middle_finger_tip_y - thumb_tip_y) ** 2) ** 0.5

                # Sprawdzenie, czy palce są blisko siebie (uszczypnięcie)
                if left_distance < 50:
                    is_left_pinching = True
                    pyautogui.click(button='left')
                else:
                    is_left_pinching = False

                if right_distance < 50:
                    is_right_pinching = True
                    pyautogui.click(button='right')
                else:
                    is_right_pinching = False

                # Przeskalowanie współrzędnych dłoni do rozdzielczości ekranu
                screen_x = screen_width - int(index_finger_tip_x * screen_width / cam_width)
                screen_y = int(index_finger_tip_y * screen_height / cam_height)

                # Wygładzanie ruchu
                smoothed_x = last_cursor_x + smooth_factor * (screen_x - last_cursor_x)
                smoothed_y = last_cursor_y + smooth_factor * (screen_y - last_cursor_y)

                # Przesunięcie kursora
                pyautogui.moveTo(smoothed_x, smoothed_y, duration=0.01)

                # Zaktualizuj ostatnią pozycję kursora
                last_cursor_x, last_cursor_y = smoothed_x, smoothed_y

        # Wyświetlenie obrazu
        cv2.imshow('Sterowanie kursorem', image)

        # Zakończenie pętli, jeśli naciśnięto klawisz 'q'
        if cv2.waitKey(1) == ord('q'):
            break

finally:
    # Zamknięcie kamery i okna
    cap.release()
    cv2.destroyAllWindows()
