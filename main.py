import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# Title of the application
st.title("Face Detection System")

# Sidebar for options
st.sidebar.header("Options")
detection_mode = st.sidebar.radio(
    "Select Detection Mode:",
    ("Upload Image", "Use Webcam")
)

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(image):
    """Detect faces in an image and draw rectangles around them."""
    # Convert the image to grayscale (required for face detection)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    # Draw rectangles around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    return image, len(faces)

if detection_mode == "Upload Image":
    st.header("Upload an Image for Face Detection")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read the image file
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Convert PIL Image to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Detect faces
        processed_image, num_faces = detect_faces(image_cv)
        
        # Convert back to PIL format for display
        processed_image_pil = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
        
        st.image(processed_image_pil, caption=f"Detected {num_faces} face(s)", use_column_width=True)
        
else:  # Webcam mode
    st.header("Real-time Face Detection using Webcam")
    st.write("Click the button below to start/stop the webcam.")
    
    # Initialize session state
    if 'run' not in st.session_state:
        st.session_state['run'] = False
    
    # Button to start/stop webcam
    start_stop = st.button("Start/Stop Webcam")
    
    if start_stop:
        st.session_state['run'] = not st.session_state['run']
    
    # Placeholder for the webcam feed
    FRAME_WINDOW = st.image([])
    
    # Webcam capture
    if st.session_state['run']:
        camera = cv2.VideoCapture(0)
        
        while st.session_state['run']:
            # Read frame from webcam
            ret, frame = camera.read()
            if not ret:
                st.error("Failed to capture frame from webcam")
                break
            
            # Detect faces
            processed_frame, num_faces = detect_faces(frame)
            
            # Convert from BGR to RGB for Streamlit display
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            # Display the processed frame
            FRAME_WINDOW.image(processed_frame_rgb)
            
            # Display number of faces detected
            st.write(f"Faces detected: {num_faces}")
        
        # Release the camera when done
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

# About section
st.sidebar.header("About")
st.sidebar.text("""
Face Detection System
Using OpenCV and Streamlit
""")