import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime
import os

# App state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "session_active" not in st.session_state:
    st.session_state.session_active = False

# Instructor login UI
def instructor_login():
    st.title("🔐 Instructor Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "admin123":
            st.session_state.authenticated = True
            st.success("Logged in successfully!")
        else:
            st.error("Invalid username or password.")

# Live dashboard UI
def dashboard():
    st.title("🎯 Live Monitoring Dashboard")
    if not st.session_state.session_active:
        if st.button("▶️ Start Monitoring"):
            st.session_state.session_active = True
            st.success("Session started")
            run_emotion_detection()
    else:
        if st.button("⏹ Stop Monitoring"):
            st.session_state.session_active = False
            st.warning("Session stopped")

# Real-time emotion detection with live bar chart beside video
def run_emotion_detection():
    video_col, graph_col = st.columns(2)  # Equal size columns

    video_placeholder = video_col.empty()
    graph_placeholder = graph_col.empty()

    log = []
    emotion_counts = Counter()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Could not open webcam.")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while st.session_state.session_active:
        ret, frame = cap.read()
        if not ret:
            st.error("⚠️ Could not read frame from webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi = frame[y:y+h, x:x+w]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            try:
                analysis = DeepFace.analyze(roi, actions=['emotion'], enforce_detection=False)
                emotion = analysis[0]['dominant_emotion']
                emotion_counts[emotion] += 1
                log.append({
                    "time": datetime.now().strftime('%H:%M:%S'),
                    "emotion": emotion
                })
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            except Exception as e:
                print(f"Error analyzing face: {e}")

        # Show webcam frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame, channels="RGB", use_container_width=True)


        # Show live emotion bar chart
        if emotion_counts:
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor('#f0f2f6')
            ax.set_facecolor('#f0f2f6')
            ax.bar(emotion_counts.keys(), emotion_counts.values(), color='skyblue')
            ax.set_ylabel("Count", color='black')
            ax.set_title("Live Emotion Distribution", color='black')
            ax.tick_params(colors='black')
            graph_placeholder.pyplot(fig)

    cap.release()
    save_log(log)

# Save session log to CSV
def save_log(log):
    if not log:
        return

    if not os.path.exists("logs"):
        os.makedirs("logs")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = f"logs/session_{timestamp}.csv"

    with open(filepath, "w") as f:
        f.write("time,emotion\n")
        for entry in log:
            f.write(f"{entry['time']},{entry['emotion']}\n")

    st.success(f"✅ Session log saved: {filepath}")

# Main app entry
def main():
    st.set_page_config("Classroom Monitoring", layout="wide")
    if not st.session_state.authenticated:
        instructor_login()
    else:
        dashboard()

if __name__ == "__main__":
    main()
