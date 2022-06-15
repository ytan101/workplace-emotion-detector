# run simple loopback for Streamlit 
import streamlit as st
from streamlit_webrtc import webrtc_streamer

from live_feed import EmotionDetection

title = st.title("Emotion Detection")
webrtc_streamer(key="example", video_processor_factory=EmotionDetection)
