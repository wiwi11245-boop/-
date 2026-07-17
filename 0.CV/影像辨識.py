import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import asyncio
import websockets
import threading
import json

# 宣告 Tasks API 模組
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='pose_landmarker_full.task'),
    running_mode=VisionRunningMode.VIDEO
)

# 共享變數與伺服器設定
current_final_action = "NONE"
CONNECTED_CLIENTS = set()

async def handler(websocket):
    """管理網頁連線的 WebSocket 處理器"""
    CONNECTED_CLIENTS.add(websocket)
    try:
        async for message in websocket:
            pass # 這裡只單向發送，不接收網頁端訊息
    except websockets.ConnectionClosed:
        pass
    finally:
        CONNECTED_CLIENTS.remove(websocket)

async def broadcast_action():
    """每秒 30 次，將當前手勢指令同步給所有連接的網頁"""
    global current_final_action
    while True:
        if CONNECTED_CLIENTS:
            message = json.dumps({"action": current_final_action})
            await asyncio.gather(*[client.send(message) for client in CONNECTED_CLIENTS], return_exceptions=True)
        await asyncio.sleep(1/30)

async def main_async_entry():
    """【修正核心】在同一個運行中的異步迴圈內同時啟動伺服器與廣播任務"""
    # 當進入此函式時，loop 已經在跑了，websockets 就不會報錯
    async with websockets.serve(handler, "localhost", 8765):
        await broadcast_action()

def start_server_thread():
    """啟動本地 WebSocket 伺服器的執行緒進入點"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    # 使用 run_until_complete 來執行我們包裝好的異步主入口
    loop.run_until_complete(main_async_entry())

# 啟動伺服器傳輸執行緒
threading.Thread(target=start_server_thread, daemon=True).start()

# ==========================================
# 影像辨識主迴圈 (保持不變)
# ==========================================
cap = cv2.VideoCapture(0)
timestamp = 0
wrist_y_history = deque(maxlen=15)
active_frames_count = 0 
last_detected_gesture = None  
gesture_streak_count = 0       
overtake_vote_pool = deque(maxlen=30)

with PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, image = cap.read()
        if not success: break
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        timestamp += 1
        results = landmarker.detect_for_video(mp_image, timestamp)
        
        instant_intent = None

        if results.pose_landmarks and len(results.pose_landmarks) > 0:
            pose_landmarks = results.pose_landmarks[0]
            try:
                h, w, _ = image.shape
                s_y, s_x = int(pose_landmarks[11].y * h), int(pose_landmarks[11].x * w)
                e_y, e_x = int(pose_landmarks[13].y * h), int(pose_landmarks[13].x * w)
                w_y, w_x = int(pose_landmarks[15].y * h), int(pose_landmarks[15].x * w)

                arm_angle = np.degrees(np.arctan2(e_y - s_y, e_x - s_x))      
                forearm_angle = np.degrees(np.arctan2(w_y - e_y, w_x - e_x))  
                
                wrist_y_history.append(w_y)
                y_movement = max(wrist_y_history) - min(wrist_y_history) if len(wrist_y_history) == 15 else 0

                if (70 < arm_angle < 110 and e_x - s_x < 50) or (w_y > s_y and abs(w_x - s_x) < 60):
                    active_frames_count = 0
                    instant_intent = "RESTING"
                else:
                    active_frames_count += 1

                if active_frames_count >= 15:
                    if -120 < forearm_angle < -35:
                        instant_intent = "RIGHT TURN"
                    elif abs(arm_angle) < 20 and not (-120 < forearm_angle < -35):
                        instant_intent = "LEFT TURN"
                    elif w_y > e_y + 20:
                        if y_movement > 12:
                            instant_intent = "DOWNWARD_ZONE"
                            overtake_vote_pool.append(1)
                        else:
                            instant_intent = "DOWNWARD_ZONE"
                            overtake_vote_pool.append(0)
            except:
                pass

        # 1秒防抖機制
        if instant_intent and instant_intent != "RESTING":
            if instant_intent == last_detected_gesture:
                gesture_streak_count += 1
            else:
                gesture_streak_count = 1  
            last_detected_gesture = instant_intent
            
            if gesture_streak_count >= 30:
                if instant_intent == "DOWNWARD_ZONE":
                    if sum(overtake_vote_pool) >= 8:
                        current_final_action = "OVERTAKE ALLOWED"
                    else:
                        current_final_action = "STOP / SLOW DOWN"
                else:
                    current_final_action = instant_intent
        else:
            gesture_streak_count = 0
            last_detected_gesture = None
            current_final_action = "NONE"
            overtake_vote_pool.clear()

        # OpenCV 視窗確認目前的辨識狀況
        cv2.putText(image, f"Server Sending: {current_final_action}", (30, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Python AI Sender', image)
        if cv2.waitKey(5) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()
