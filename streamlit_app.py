import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode
import av
import math
import logging

# 初始化日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# WebRTC 配置
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class HandGestureTransformer(VideoTransformerBase):
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.tip_ids = [4, 8, 12, 16, 20]
        self.vol_history = []
        self.smooth_factor = 5
        self.min_dist = 30
        self.max_dist = 300
        self.volume = 50  # 默认音量

    def transform(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            img = cv2.resize(img, (640, 480))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    # 获取关键点坐标
                    lm_list = []
                    for id, lm in enumerate(hand_landmarks.landmark):
                        h, w, c = img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lm_list.append([id, cx, cy])

                    # 计算拇指和食指距离
                    if len(lm_list) >= 9:
                        x1, y1 = lm_list[4][1], lm_list[4][2]
                        x2, y2 = lm_list[8][1], lm_list[8][2]
                        length = math.hypot(x2 - x1, y2 - y1)

                        # 平滑处理
                        vol = np.interp(length, [self.min_dist, self.max_dist], [0, 100])
                        self.vol_history.append(vol)
                        if len(self.vol_history) > self.smooth_factor:
                            self.vol_history.pop(0)
                        self.volume = sum(self.vol_history) / len(self.vol_history)

            # 显示音量信息
            cv2.putText(img, f'Volume: {int(self.volume)}%', (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 绘制音量条
            bar_width, bar_height = 25, 150
            bar_x, bar_y = 20, 60
            vol_fill_height = int(np.interp(self.volume, [0, 100], [0, bar_height]))
            
            cv2.rectangle(img, (bar_x, bar_y),
                         (bar_x + bar_width, bar_y + bar_height),
                         (0, 255, 0), 1)
            cv2.rectangle(img, (bar_x, bar_y + bar_height - vol_fill_height),
                         (bar_x + bar_width, bar_y + bar_height),
                         (0, 255, 0), cv2.FILLED)

            return img
        except Exception as e:
            logging.error(f"视频处理错误: {str(e)}")
            return frame

def main():
    st.title("手势控制音量演示")
    st.markdown("""
    ### 使用说明：
    1. 允许浏览器访问摄像头。
    2. 手势控制：
       - 👆 拇指和食指分开：增大音量
       - 🤏 拇指和食指靠近：减小音量
    *注意：此演示显示音量控制界面，实际音量控制需要在本地环境中实现。*
    """)

    # 初始化WebRTC流
    webrtc_ctx = webrtc_streamer(
        key="hand-gesture",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=HandGestureTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if st.button("重置应用"):
        st.experimental_rerun()

if __name__ == "__main__":
    main()
