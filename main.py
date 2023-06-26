import cv2
import time
import imageio
import mediapipe as mp
import numpy as np
from sklearn.linear_model import LogisticRegression
anh1_path = "anhnen.jpg"
anh2_path = "anhtrain.jpg"
anhnen_path = "anhthaynen.jpg"
anh1 = cv2.imread(anh1_path)
anh2 = cv2.imread(anh2_path)
anhnen = cv2.imread(anhnen_path)
anh1 = cv2.resize(anh1, (640, 480))
anh2 = cv2.resize(anh2, (640, 480))
anhnen = cv2.resize(anhnen, (640,480))
x = np.concatenate((anh1.reshape(-1, 3), anh2.reshape(-1, 3)), axis=0)
y = np.concatenate((np.zeros(anh1.shape[0] * anh1.shape[1]), np.ones(anh2.shape[0] * anh2.shape[1])))
class_weights = {0: 1, 1: 2} 
model_chuyennen = LogisticRegression(class_weight=class_weights)
model_chuyennen.fit(x, y)
def chuyennen(model,frame):
    frame = cv2.resize(frame, (640, 480))
    x_test = frame.reshape(-1, 3)
    y_pred = model.predict(x_test)
    
    output_image = np.where(y_pred.reshape(frame.shape[0], frame.shape[1], 1) == 0, anhnen, frame)
    output_image = output_image.astype(np.uint8)
    return output_image

# Hàm để thêm mặt nạ
def add_mask(image, resized_mask, face_bbox):
    x, y, w, h = face_bbox
    if x-50<= 0 or y-50<=0 or x+w >= 640 or y+h >=480:
        # print("loiday")
        return image 
    else:
    # resized_mask = cv2.cvtColor(resized_mask, cv2.COLOR_RGBA2BGR)
        resized_mask = cv2.resize(resized_mask, (w+50, h+50))
        # mask_gray = cv2.cvtColor(resized_mask, cv2.COLOR_RGB2GRAY)
        # output = image[y:y+h, x:x+w].copy()
        if y>50 and x>30 and x+w+20 <640 and y+h <480:
            image[y-50:y+h, x-30:x+w+20][resized_mask != (255,255,255)] = resized_mask[resized_mask != (255,255,255)]
        return image

# # Hàm lấy tọa độ khuôn mặt
def detect_face(image):
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    pad = 60
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image_rgb)
        x = 0
        y = 0
        w = 0
        h = 0
        if results.detections:
            for detection in results.detections:
                bbox_cordinates = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                x, y, w, h = int(bbox_cordinates.xmin * iw), int(bbox_cordinates.ymin * ih), \
                             int(bbox_cordinates.width * iw), int(bbox_cordinates.height * ih)
                # x-=pad-20
                # y-=pad+15
                # w+=pad
                # h+=pad
                # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return image, x, y ,w, h




# Đọc video và resize video theo tỉ lệ
def video_to_frame_list(video_path, tile):
    frame_list = []
    # Đọc video
    video = cv2.VideoCapture(video_path)
    # Kiểm tra xem video có tồn tại hay không
    if not video.isOpened():
        raise Exception("Không thể mở video")
    # Lặp qua từng khung hình trong video và thêm vào danh sách
    while True:
        ret, frame = video.read()
        # Kiểm tra xem đã đọc hết video hay chưa
        if not ret:
            break
        w_n = int(frame.shape[1]*tile)
        h_n = int(frame.shape[0]*tile)
        frame = cv2.resize(frame, (w_n, h_n))
        # Thêm khung hình vào danh sách
        frame_list.append(frame)

    # Giải phóng tài nguyên video
    video.release()
    return frame_list, h_n, w_n
#Đọc effect và mặt ạ và resize

effect_gif_path = 'vongtronmathuat2crop.mp4'
vongtron_path = 'hieuungvongtron2.mp4'
nutdo_path = 'nutdo1.png'
matna_path = 'mat_drstrange.jpg'
matna = cv2.imread(matna_path)
nutdo = cv2.imread(nutdo_path)
nutdo = cv2.resize(nutdo, (50,50))


nutdo_height = nutdo.shape[0]
nutdo_width = nutdo.shape[1]

cap = cv2.VideoCapture(0)

while not cap.isOpened():
    pass

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Đọc thư viện
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


# Chuyển đổi video thành danh sách khung hình
tile = 0.15   #tỉ lệ
effect_gif, h_e, w_e = video_to_frame_list(effect_gif_path, tile)

vongtron, h_vt, w_vt = video_to_frame_list(vongtron_path, 1)
framevongtron = [frame_tron for frame_tron in vongtron]
bgs = [frame_gif for frame_gif in effect_gif]
i = 0
j = 0
sizes = len(effect_gif)
size_vongtron = len(vongtron)
cuchitay = np.zeros((frame_height,frame_width))

w_n =  int(w_e/2)
h_n = int(h_e/2)

flag = 0
while True:
    # Đọc frame từ video capture
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Phát hiện và xử lý pose trong frame
    results = pose.process(image_rgb)
    nutdo_gray = cv2.cvtColor(nutdo, cv2.COLOR_RGB2GRAY)

    if not ret:
        break

    #Hiển thị nút khi chưa được bấm
    # if flag == 0:
        # frame[240-int(nutdo_height/2):240+int(nutdo_height/2),120-int(nutdo_width/2):120+int(nutdo_width/2) ][nutdo_gray!=255] = nutdo[nutdo_gray!=255]

    x17 = 0
    x18 = 0
    y17 = 0
    y18 = 0
    #Đọc vị trí trên bàn tay
    if results.pose_landmarks:

        landmark_17 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX]
        landmark_18 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX]
        image_height, image_width, _ = frame.shape
        x17, y17 = int(landmark_17.x * image_width), int(landmark_17.y * image_height)
        x18, y18 = int(landmark_18.x * image_width), int(landmark_18.y * image_height)

    # if x17 < 120 and x17 > 80 and y17 < 240 and y17 > 200:  # Nếu tay phải chạm nút
    #     flag = 1
    if x17 != 0 and x17 != 0 and y17 != 0 and y17 != 0:  # Nếu tay phải chạm nút
        if abs(x17-x18) <10 and abs(y17 - y18) < 10:
            flag = 2
    if flag == 2:
        if j<size_vongtron:
            image_with_face, x, y ,w ,h= detect_face(frame)
            # Lấy vị trí và thêm khuôn mặt
            
            frame = add_mask(frame, matna, (x, y, w, h))
            bgsi = framevongtron[j]
            bgsi = cv2.resize(bgsi, (640,480))
            
            bgsi = bgsi.astype(np.uint8)
            frame = cv2.addWeighted(frame, 1, bgsi, 1.5, 0)
            j +=1
        else:
            image_trang = np.zeros((480, 640, 3), dtype=np.uint8)
            frame = image_trang
            flag =1

    if flag == 1:
        image_with_face, x, y ,w ,h= detect_face(frame)
        frame = chuyennen(model_chuyennen, frame)
        # Lấy vị trí và thêm khuôn mặt
        
        frame = add_mask(frame, matna, (x, y, w, h))
        # if x!=0:
        

        # Thêm từng frame effect vào từng frame lấy từ camera cho đến khi hết effect
        if i < sizes:
            if x17-int(w_e/2) <= 0 or x17 + int(w_e/2) >= 640 or y17-int(h_e/2) <= 0 or y17+int(h_e/2) >=480:
                frame = frame
            elif x18-int(w_e/2) <= 0 or x18 + int(w_e/2) >= 640 or y18-int(h_e/2) <= 0 or y18+int(h_e/2) >=480:
                frame = frame

            else:

                bgsi = bgs[i]
                gray_ef = cv2.cvtColor(bgsi, cv2.COLOR_BGR2GRAY)
         
                bgsi_n1 = np.full((480, 640, 3), 0)
                bgsi_n1[y17-int(h_e/2): y17+int(h_e/2), x17-int(w_e/2): x17+int(w_e/2), ] = bgsi
                bgsi1 = bgsi_n1.astype(np.uint8)

                bgsi_n2 = np.full((480, 640, 3), 0)
                bgsi_n2[y18-int(h_e/2): y18+int(h_e/2), x18-int(w_e/2): x18+int(w_e/2), ] = bgsi
                bgsi2 = bgsi_n2.astype(np.uint8)
                

                frame = cv2.addWeighted(frame, 1, bgsi1, 1.5, 0)
                frame = cv2.addWeighted(frame, 1, bgsi2, 1.5, 0)

                
                i += 1
        
        else:
            i = 0
            # flag = 0

        
    frame = cv2.resize(frame, (640, 480))

    cv2.imshow('MediaPipe Pose', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
