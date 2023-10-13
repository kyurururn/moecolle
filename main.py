import requests
import qrcode
import dropbox
import cv2
import numpy as np
import mediapipe as mp
import math
import time

def upload():
    ACCESS_TOKEN = "<ACCESS_TOKEN>"
    WEBHOOK_URL = "<WEBHOOK_URL>"

    time.sleep(5)

    with open("num.txt","r") as file:
        lines = file.readlines()

    with open("num.txt","w") as file:
        file.write(str(int(lines[0]) + 1))

    local_file_path = "capture.jpg"
    dropbox_file_path = f"/{lines[0]}.jpg"

    dbx = dropbox.Dropbox(ACCESS_TOKEN)
    with open(local_file_path, 'rb') as f:
        dbx.files_upload(f.read(), dropbox_file_path)
    shared_link = dbx.sharing_create_shared_link(dropbox_file_path)
    print('Shared link for the file:', shared_link.url)

    urll = shared_link.url
    urll = urll[:-1]

    content = "QR"

    img = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )

    img.add_data(urll)
    img.make(fit=True)

    img = img.make_image(fill_color="black", back_color="white")
    img.save('result.png')

    with open('result.png', "rb") as file:
        files = {"file":file}
        payload = {"content":content}
        responce = requests.post(WEBHOOK_URL,data=payload,files=files)



def chroma_keying(frame, background_image, lower_color, upper_color):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, lower_color, upper_color)
    mask_inverse = cv2.bitwise_not(mask)
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask_inverse)
    masked_background = cv2.bitwise_and(background_image, background_image, mask=mask)
    result_frame = cv2.add(masked_frame, masked_background)
    return result_frame

def intersect(p1, p2, p3, p4):
    tc1 = (p1[0] - p2[0]) * (p3[1] - p1[1]) + (p1[1] - p2[1]) * (p1[0] - p3[0])
    tc2 = (p1[0] - p2[0]) * (p4[1] - p1[1]) + (p1[1] - p2[1]) * (p1[0] - p4[0])
    td1 = (p3[0] - p4[0]) * (p1[1] - p3[1]) + (p3[1] - p4[1]) * (p3[0] - p1[0])
    td2 = (p3[0] - p4[0]) * (p2[1] - p3[1]) + (p3[1] - p4[1]) * (p3[0] - p2[0])
    return tc1*tc2<0 and td1*td2<0

def distance_2points(p1, p2, p3, p4):
    distance = math.sqrt((p1 - p3)**2 + (p2 - p4)**2)
    return distance


cap = cv2.VideoCapture(0)

background_image = cv2.imread('background.jpg')
background_image = cv2.resize(background_image,(640,480))

background_image_a = cv2.imread('background_a.jpg')
background_image_a = cv2.resize(background_image_a,(640,480))

lower_green = np.array([30, 70, 70])
upper_green = np.array([90, 255, 255])

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(model_complexity=0,min_detection_confidence=0.5,min_tracking_confidence=0.5)

while True:
    ret, frame = cap.read()

    frame.flags.writeable = False
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame)

    # 検出された手の骨格をカメラ画像に重ねて描画
    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    finger_check = False
    peace_check  = False
    heart_check  = False
    big_heart_check  = False
    ear_check = False
    if results.multi_hand_landmarks:
        if len(results.multi_hand_landmarks) >= 2:
            d1 = distance_2points(results.multi_hand_landmarks[0].landmark[8].x, results.multi_hand_landmarks[0].landmark[8].y, results.multi_hand_landmarks[1].landmark[8].x, results.multi_hand_landmarks[1].landmark[8].y)
            d2 = distance_2points(results.multi_hand_landmarks[0].landmark[12].x, results.multi_hand_landmarks[0].landmark[12].y, results.multi_hand_landmarks[1].landmark[12].x, results.multi_hand_landmarks[1].landmark[12].y)
            
            if d1 <= 0.15 and d2 <= 0.15:
                heart_check = True
            
            d3 = distance_2points(results.multi_hand_landmarks[0].landmark[20].x, results.multi_hand_landmarks[0].landmark[20].y, results.multi_hand_landmarks[1].landmark[20].x, results.multi_hand_landmarks[1].landmark[20].y)
            d4 = distance_2points(results.multi_hand_landmarks[0].landmark[4].x, results.multi_hand_landmarks[0].landmark[4].y, results.multi_hand_landmarks[1].landmark[4].x, results.multi_hand_landmarks[1].landmark[4].y)
            
            if d3 <= 0.15 and d4 <= 0.15:
                big_heart_check = True

            d5 = distance_2points(results.multi_hand_landmarks[0].landmark[8].x, results.multi_hand_landmarks[0].landmark[8].y, results.multi_hand_landmarks[0].landmark[12].x, results.multi_hand_landmarks[0].landmark[12].y)
            d6 = distance_2points(results.multi_hand_landmarks[0].landmark[12].x, results.multi_hand_landmarks[0].landmark[12].y, results.multi_hand_landmarks[0].landmark[16].x, results.multi_hand_landmarks[0].landmark[16].y)
            d7 = distance_2points(results.multi_hand_landmarks[1].landmark[8].x, results.multi_hand_landmarks[1].landmark[8].y, results.multi_hand_landmarks[1].landmark[12].x, results.multi_hand_landmarks[1].landmark[12].y)
            d8 = distance_2points(results.multi_hand_landmarks[1].landmark[12].x, results.multi_hand_landmarks[1].landmark[12].y, results.multi_hand_landmarks[1].landmark[16].x, results.multi_hand_landmarks[1].landmark[16].y)
            
            print(d5)

            if d5 <= 0.2 and d6 <= 0.2 and d7 <= 0.2 and d8 <= 0.2:
                ear_check = True

        for hand_landmarks in results.multi_hand_landmarks:
            first_landmark = hand_landmarks.landmark[0]
            x, y, z = first_landmark.x, first_landmark.y, first_landmark.z

            p1 = [hand_landmarks.landmark[2].x,hand_landmarks.landmark[2].y]
            p2 = [hand_landmarks.landmark[4].x,hand_landmarks.landmark[4].y]
            p3 = [hand_landmarks.landmark[6].x,hand_landmarks.landmark[6].y]
            p4 = [hand_landmarks.landmark[8].x,hand_landmarks.landmark[8].y]

            finger_check =  finger_check or (intersect(p1,p2,p3,p4) and hand_landmarks.landmark[4].y <= hand_landmarks.landmark[2].y and hand_landmarks.landmark[8].y <= hand_landmarks.landmark[6].y)

            peace_check = peace_check or ((hand_landmarks.landmark[8].y <= hand_landmarks.landmark[7].y <= hand_landmarks.landmark[6].y <= hand_landmarks.landmark[5].y) and (hand_landmarks.landmark[4].y <= hand_landmarks.landmark[3].y <= hand_landmarks.landmark[2].y <= hand_landmarks.landmark[1].y))

            #mp_drawing.draw_landmarks(
            #    frame,
            #    hand_landmarks,
            #    mp_hands.HAND_CONNECTIONS,
            #    mp_drawing_styles.get_default_hand_landmarks_style(),
            #    mp_drawing_styles.get_default_hand_connections_style())q
  
    frame = cv2.resize(frame,(640,480))

    if finger_check:   
        result_frame = chroma_keying(frame, background_image_a, lower_green, upper_green)
        cv2.putText(result_frame,
                    text='yubi_heart',
                    org=(20, 20),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.0,
                    color=(0, 255, 0),
                    thickness=2,
                    lineType=cv2.LINE_4)
    elif peace_check:
        result_frame = chroma_keying(frame, background_image_a, lower_green, upper_green)
        
        cv2.putText(result_frame,
                    text='peace',
                    org=(20, 20),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.0,
                    color=(0, 255, 0),
                    thickness=2,
                    lineType=cv2.LINE_4)
    elif big_heart_check:
        result_frame = chroma_keying(frame, background_image_a, lower_green, upper_green)
        
        cv2.putText(result_frame,
                    text='big heart',
                    org=(20, 20),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.0,
                    color=(0, 255, 0),
                    thickness=2,
                    lineType=cv2.LINE_4)
    elif heart_check:
        result_frame = chroma_keying(frame, background_image_a, lower_green, upper_green)
        
        cv2.putText(result_frame,
                    text='heart',
                    org=(20, 20),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.0,
                    color=(0, 255, 0),
                    thickness=2,
                    lineType=cv2.LINE_4)
    else:
        result_frame = chroma_keying(frame, background_image, lower_green, upper_green)
        
        cv2.putText(result_frame,
                    text='none',
                    org=(20, 20),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.0,
                    color=(0, 255, 0),
                    thickness=2,
                    lineType=cv2.LINE_4)

    cv2.imshow('Chroma Keying', result_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):  # Check for the space key press
        cv2.imwrite('capture.jpg', result_frame)
        time.sleep(5)
        upload()




cap.release()
cv2.destroyAllWindows()