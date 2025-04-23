import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Title of the application
st.title("Face Detection System")

# Sidebar for options
st.sidebar.header("Options")
detection_mode = st.sidebar.radio(
    "Select Detection Mode:",
    ("Upload Image", "Use Webcam")
)

# Load the pre-trained Haar Cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(image):
    """Detect faces in an image and draw rectangles around them."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    return image, len(faces)

if detection_mode == "Upload Image":
    st.header("Upload an Image for Face Detection")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        processed_image, num_faces = detect_faces(image_cv)
        processed_image_pil = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
        
        st.image(processed_image_pil, caption=f"Detected {num_faces} face(s)", use_container_width=True)
        
else:  # Webcam mode
    st.header("Real-time Face Detection using Webcam")
    st.write("Click the button below to start/stop the webcam.")
    
    if 'run' not in st.session_state:
        st.session_state['run'] = False
    
    if st.button("Start/Stop Webcam"):
        st.session_state['run'] = not st.session_state['run']
    
    FRAME_WINDOW = st.image([])
    
    if st.session_state['run']:
        camera = cv2.VideoCapture(0)
        
        while st.session_state['run']:
            ret, frame = camera.read()
            if not ret:
                st.error("Failed to capture frame from webcam")
                break
            
            processed_frame, num_faces = detect_faces(frame)
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(processed_frame_rgb)
            st.write(f"Faces detected: {num_faces}")
        
        camera.release()
    else:
        st.write("Webcam is stopped")

# Instructions
st.sidebar.header("Instructions")
st.sidebar.info("""
1. Select detection mode (Upload Image or Use Webcam)
2. For image upload: Select an image file
3. For webcam: Click the Start/Stop button
""")