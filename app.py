import threading
import cv2
import av
import streamlit as st
from matplotlib import pyplot as plt
from streamlit_webrtc import webrtc_streamer

# 1) Cấu hình trang Streamlit
st.set_page_config(page_title="Ultralytics Streamlit App", layout="wide")

# 2) ICE servers (STUN + TURN) gán cứng từ Metered
ice_servers = [
    {"urls": ["stun:stun.relay.metered.ca:80"]},
    {
        "urls": ["turn:global.relay.metered.ca:80"],
        "username": "d7c5c4f386501ce1a4b593e4",
        "credential": "bnRzs/8viqUMSVaJ",
    },
    {
        "urls": ["turn:global.relay.metered.ca:80?transport=tcp"],
        "username": "d7c5c4f386501ce1a4b593e4",
        "credential": "bnRzs/8viqUMSVaJ",
    },
    {
        "urls": ["turn:global.relay.metered.ca:443"],
        "username": "d7c5c4f386501ce1a4b593e4",
        "credential": "bnRzs/8viqUMSVaJ",
    },
    {
        "urls": ["turns:global.relay.metered.ca:443?transport=tcp"],
        "username": "d7c5c4f386501ce1a4b593e4",
        "credential": "bnRzs/8viqUMSVaJ",
    },
    # (tuỳ chọn) fallback STUN của Google
    {"urls": ["stun:stun.l.google.com:19302"]},
]

# 3) Tạo layout 2 cột
col1, col2 = st.columns(2)
lock = threading.Lock()
img_container = {"img": None}

# 4) Callback để xử lý mỗi frame và lưu lại ảnh gốc
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    with lock:
        img_container["img"] = img
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# 5) Khởi tạo WebRTC streamer với ICE servers đã định nghĩa
with col1:
    ctx = webrtc_streamer(
        key="example",
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={"iceServers": ice_servers},
        async_processing=True,
    )

# 6) Thiết lập vẽ histogram ở cột thứ hai
fig_place = col2.empty()
fig, ax = plt.subplots()

# 7) Vòng lặp cập nhật histogram liên tục trong khi video đang phát
if ctx and ctx.state.playing:
    while ctx.state.playing:
        with lock:
            img = img_container["img"]
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ax.cla()
        ax.hist(gray.ravel(), bins=256, range=[0, 256])
        fig_place.pyplot(fig)
