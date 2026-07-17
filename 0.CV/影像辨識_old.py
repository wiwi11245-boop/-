import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# 1. 宣告 Tasks API 模組
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='pose_landmarker_full.task'),
    running_mode=VisionRunningMode.VIDEO
)

# 歷史紀錄佇列 (15幀 = 約0.5秒)
wrist_y_history = deque(maxlen=15)
active_frames_count = 0 

# 1秒穩定度判定變數
last_detected_gesture = None  
gesture_streak_count = 0       
final_action = None           

# ====== 新增：用於動態超車的二階段決策佇列 ======
# 記錄這 1 秒（30幀）內，到底有多少幀被判定為有晃動
overtake_vote_pool = deque(maxlen=30)

cap = cv2.VideoCapture(0)
timestamp = 0

print(">>> 交通手勢【動態晃動優化版】已啟動...")

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
                
                cv2.circle(image, (s_x, s_y), 8, (255, 0, 0), -1) 
                cv2.circle(image, (e_x, e_y), 8, (0, 255, 0), -1) 
                cv2.circle(image, (w_x, w_y), 8, (0, 0, 255), -1) 

                arm_angle = np.degrees(np.arctan2(e_y - s_y, e_x - s_x))      
                forearm_angle = np.degrees(np.arctan2(w_y - e_y, w_x - e_x))  
                
                # Y 軸晃動度
                wrist_y_history.append(w_y)
                y_movement = max(wrist_y_history) - min(wrist_y_history) if len(wrist_y_history) == 15 else 0

                # ====== 休息狀態判斷 ======
                if (70 < arm_angle < 110 and e_x - s_x < 50) or (w_y > s_y and abs(w_x - s_x) < 60):
                    active_frames_count = 0
                    instant_intent = "RESTING"
                else:
                    active_frames_count += 1

                # ====== 核心手勢判定 (離開休息區滿 0.5 秒後啟用) ======
                if active_frames_count >= 15:
                    
                    # 1. 右轉彎
                    if -120 < forearm_angle < -35:
                        instant_intent = "RIGHT TURN"
                    
                    # 2. 左轉彎
                    elif abs(arm_angle) < 20 and not (-120 < forearm_angle < -35):
                        instant_intent = "LEFT TURN"
                        
                    # 3. 往下沈區間 (將減速與超車合併為同一大類家族，解決跳動問題)
                    elif w_y > e_y + 20:
                        # 記錄這一影格有沒有晃動趨勢 (1代表晃動，0代表靜止)
                        if y_movement > 12:
                            instant_intent = "DOWNWARD_ZONE"
                            overtake_vote_pool.append(1)
                        else:
                            instant_intent = "DOWNWARD_ZONE"
                            overtake_vote_pool.append(0)

            except Exception as e:
                pass

        # ====== 🛠️ 修正版：1秒穩定度決策機制 ======
        if instant_intent and instant_intent != "RESTING":
            # 如果手勢相同，就累加時間
            if instant_intent == last_detected_gesture:
                gesture_streak_count += 1
            else:
                gesture_streak_count = 1  
            
            last_detected_gesture = instant_intent
            
            # 當手勢群組成功維持滿 1 秒 (30幀)
            if gesture_streak_count >= 30:
                if instant_intent == "DOWNWARD_ZONE":
                    # 二階段決策：看過去這段時間內，舉手晃動的比例高不高
                    # 如果 30 幀內有超過 8 幀以上都在晃動，就判定為「允許超車」，否則就是「減速暫停」
                    if sum(overtake_vote_pool) >= 8:
                        final_action = "OVERTAKE ALLOWED"
                    else:
                        final_action = "STOP / SLOW DOWN"
                else:
                    final_action = instant_intent
        else:
            gesture_streak_count = 0
            last_detected_gesture = None
            final_action = None
            overtake_vote_pool.clear() # 休息時清空投票池

        # 5. 顯示確認後的最終指令
        if final_action:
            display_text = f"Action: {final_action} (STABLE)"
            cv2.putText(image, display_text, (30, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(image, "Action: NONE (HOLD GESTURE 1s)", (30, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('Traffic Gesture - Engine V5', image)
        if cv2.waitKey(5) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()
