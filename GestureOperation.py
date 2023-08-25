import cv2
import mediapipe as mp
import pyautogui
import math

mouse_down_flag = False

# パソコンの画面サイズを取得
height = pyautogui.size().height
width = pyautogui.size().width

# マウスカーソルの移動粒度
granularity = 20

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
 
# Webカメラから入力
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
 
    # 検出された手の骨格をカメラ画像に重ねて描画
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:

        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        # 親指の座標を取得
        thumb_x = results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.THUMB_TIP].x
        thumb_y = results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.THUMB_TIP].y
        index_finger_x = results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
        index_finger_y = results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
        
        # 人差し指と親指の間の距離を計算
        distance = math.sqrt((thumb_x - index_finger_x)**2 + (thumb_y - index_finger_y)**2)
        print(distance)

        mouse_x = math.ceil((width - (thumb_x + index_finger_x)/2*width)/granularity)*granularity
        mouse_y = math.ceil((thumb_y + index_finger_y)/2*height/granularity)*granularity

        # 距離によってマウスの左クリックを押す
        if distance < 0.08:
            pyautogui.moveTo(mouse_x, mouse_y, duration=0)
            if not mouse_down_flag:
                pyautogui.mouseDown()
                mouse_down_flag = True
        # 距離によってマウスの左クリックを離す
        else:
            pyautogui.moveTo(mouse_x, mouse_y, duration=0)
            if mouse_down_flag:
                pyautogui.mouseUp()
                mouse_down_flag = False
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))

    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()