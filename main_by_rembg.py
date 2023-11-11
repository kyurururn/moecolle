import cv2
import threading
import time
import dropbox
from PIL import Image
from rembg import remove
import qrcode
import requests
import numpy as np
import mediapipe as mp
import math

HEIGHT = 960
WIDTH  = 1280
LOCAL_FILEPATH = "capture.jpg"
ACCESS_TOKEN = "<ACCESS_TOKEN>"
WEBHOOK_URL  = "<WEBHOOK_URL>"

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH ,HEIGHT)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,WIDTH)

mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands          = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh      = mp.solutions.face_mesh
mp_hands          = mp.solutions.hands

hands = mp_hands.Hands(model_complexity=0,min_detection_confidence=0.5,min_tracking_confidence=0.5)

def intersect(p1, p2, p3, p4):
    tc1 = (p1[0] - p2[0]) * (p3[1] - p1[1]) + (p1[1] - p2[1]) * (p1[0] - p3[0])
    tc2 = (p1[0] - p2[0]) * (p4[1] - p1[1]) + (p1[1] - p2[1]) * (p1[0] - p4[0])
    td1 = (p3[0] - p4[0]) * (p1[1] - p3[1]) + (p3[1] - p4[1]) * (p3[0] - p1[0])
    td2 = (p3[0] - p4[0]) * (p2[1] - p3[1]) + (p3[1] - p4[1]) * (p3[0] - p2[0])
    return tc1*tc2<0 and td1*td2<0

def distance_2points(p1, p2, p3, p4):
    distance = math.sqrt((p1 - p3)**2 + (p2 - p4)**2)
    return distance

def pause_check(frame):
    frame.flags.writeable = False
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame)

    finger_check    = False
    peace_check     = False
    heart_check     = False
    big_heart_check = False
    
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

    if   finger_check:    return "finger"
    elif peace_check:     return "peace"
    elif big_heart_check: return "bigheart"
    elif heart_check:     return "heart"
    else: return "random"
    
def upload():
    time.sleep(1)

    with open("num.txt","r") as file:
        lines = file.readlines()
    
    with open("num.txt","w") as file:
        file.write(str(int(lines[0]) + 1))
    
    dropbox_filepath = f"/{lines[0]}.jpg"
    dbx = dropbox.Dropbox(ACCESS_TOKEN)
    with open(LOCAL_FILEPATH, "rb") as f:
        dbx.files_upload(f.read(), dropbox_filepath)
    shared_link = dbx.sharing_create_shared_link(dropbox_filepath)
    print("Shared link for the file:", shared_link.url)

    url = shared_link.url
    url = url[:-1]
    url += "0&raw=1"

    img = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )

    img.add_data(url)
    img.make(fit=True)
    img = img.make_image(fill_color="black",back_color="white")
    img.save('qr.png')

    with open("qr.png","rb") as file:
        files = {"file":file}
        payload = {"content":"QR"}
        responce = requests.post(WEBHOOK_URL,data=payload,files=files)

def process_image(img):
    time.sleep(3)
    ret, frame = cap.read()
    if ret:
        pause = pause_check(frame=frame)
        input_img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        input_img = Image.fromarray(input_img)
        output_img = remove(input_img)

        background_img = Image.open("background.png")
        background_img = background_img.resize(output_img.size)
        background_img = background_img.convert("RGBA")
        output_img = output_img.convert("RGBA")

        combined_img = Image.alpha_composite(background_img, output_img)
        combined_img = combined_img.convert('RGB')
        combined_img.save(LOCAL_FILEPATH)

        time.sleep(1)
        upload()

        #cv2.imwrite('result.jpg', frame)
        #print("画像が保存されました")

while True:
    ret,frame = cap.read()
    if not ret: break

    final_frame = cv2.resize(frame, (int(WIDTH / 2), int(HEIGHT / 2)))
    cv2.imshow("final",final_frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord("q"): break
    elif key & 0xFF == ord(" "):
        threading.Thread(target=process_image, args=(frame,)).start()

cap.release()
cv2.destroyAllWindows() 
