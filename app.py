import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

st.set_page_config(layout="wide")
st.title("MediaPipe Pose - World Landmarks Detection")

# Initialize session state
if 'run_detection' not in st.session_state:
    st.session_state.run_detection = False

# Control buttons
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 4])
with col_btn1:
    if st.button("▶️ Start", width="stretch"):
        st.session_state.run_detection = True
with col_btn2:
    if st.button("⏹️ Stop", width="stretch"):
        st.session_state.run_detection = False

st.divider()

# Create two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Live Feed")
    frame_placeholder1 = st.empty()

with col2:
    st.subheader("Detected Landmarks")
    frame_placeholder2 = st.empty()

fps_placeholder = st.empty()

# Initialize webcam only if detection is running
if not st.session_state.run_detection:
    st.info("Click 'Start' to begin pose detection")
    st.stop()

cap = cv2.VideoCapture(0)

# Pose detection
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1
) as pose:
    
    prev_time = time.time()
    
    while cap.isOpened() and st.session_state.run_detection:
        success, frame = cap.read()
        if not success:
            st.warning("Failed to read from webcam")
            break
        
        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        
        # Convert BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Process the image
        results = pose.process(image)
        
        # Draw on the image
        image.flags.writeable = True
        image_with_landmarks = image.copy()
        
        if results.pose_landmarks:
            # Draw landmarks on the image
            mp_drawing.draw_landmarks(
                image_with_landmarks,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # Display world landmarks info
            if results.pose_world_landmarks:
                # Add FPS and world landmark count
                cv2.putText(
                    image_with_landmarks,
                    f"FPS: {fps:.2f} | World Landmarks: {len(results.pose_world_landmarks.landmark)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
        
        # Display frames
        frame_placeholder1.image(image, channels="RGB", width="stretch")
        frame_placeholder2.image(image_with_landmarks, channels="RGB", width="stretch")
        
        # Update FPS
        fps_placeholder.metric("FPS", f"{fps:.2f}")
        
        # Break on 'q' key (won't work in Streamlit but keeps the loop controlled)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
