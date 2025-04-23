import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time

# Title of the application
st.title("Face Detection System")

# Sidebar for options
st.sidebar.header("Options")
detection_mode = st.sidebar.radio(
    "Select Detection Mode:",
    ("Upload Image", "Use Webcam")
)

# Initialize session state variables
if 'camera_enabled' not in st.session_state:
    st.session_state['camera_enabled'] = False
if 'frame_count' not in st.session_state:
    st.session_state['frame_count'] = 0

# Load the pre-trained Haar Cascade classifier
@st.cache_resource
def load_cascade_classifier():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

face_cascade = load_cascade_classifier()

def detect_faces(image):
    """Detect faces in an image and draw rectangles around them."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    return image, len(faces)

def toggle_camera():
    st.session_state['camera_enabled'] = not st.session_state['camera_enabled']
    st.session_state['frame_count'] = 0

if detection_mode == "Upload Image":
    st.header("Upload an Image for Face Detection")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    # Add some sliders for face detection parameters
    st.sidebar.subheader("Detection Parameters")
    scale_factor = st.sidebar.slider("Scale Factor", 1.05, 1.5, 1.1, 0.05)
    min_neighbors = st.sidebar.slider("Min Neighbors", 1, 10, 5, 1)
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Convert PIL Image to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Get parameters and detect faces
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(30, 30)
        )
        
        # Draw rectangles and display
        for (x, y, w, h) in faces:
            cv2.rectangle(image_cv, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        processed_image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        st.image(processed_image_rgb, caption=f"Detected {len(faces)} face(s)", use_container_width=True)
        
else:  # Webcam mode
    st.header("Real-time Face Detection using Webcam")
    
    # Camera control button
    st.button("Start/Stop Webcam", on_click=toggle_camera)
    
    # Frame placeholder
    frame_placeholder = st.empty()
    status_text = st.empty()
    
    # Webcam handling
    if st.session_state['camera_enabled']:
        status_text.text("Starting camera... Please allow access if prompted.")
        camera = cv2.VideoCapture(0)
        
        if not camera.isOpened():
            status_text.error("Could not access webcam. Please check your connections and permissions.")
            st.session_state['camera_enabled'] = False
        else:
            status_text.text("Camera is running. Click 'Start/Stop Webcam' to stop.")
            
            # Capture a single frame per Streamlit run
            ret, frame = camera.read()
            if ret:
                st.session_state['frame_count'] += 1
                processed_frame, num_faces = detect_faces(frame)
                frame_placeholder.image(
                    cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB),
                    caption=f"Faces detected: {num_faces} (Frame {st.session_state['frame_count']})",
                    use_container_width=True
                )
            else:
                status_text.error("Failed to capture frame")
            
            camera.release()
            
            # Auto-rerun to create animation effect
            time.sleep(0.1)  # Small delay to control frame rate
            st.experimental_rerun()
    else:
        status_text.text("Webcam is stopped. Click 'Start/Stop Webcam' to begin.")

# Instructions
st.sidebar.header("Instructions")
st.sidebar.info("""
1. Select detection mode (Upload Image or Use Webcam)
2. For image upload: Select an image file
3. For webcam: Click the Start/Stop button
4. Adjust detection parameters in the sidebar when using image upload mode
""")

# Add information about the application
st.sidebar.header("About")
st.sidebar.info("""
This application uses OpenCV's Haar Cascade classifier to detect faces in images or webcam streams.

The face detection algorithm looks for patterns in the image that resemble human faces based on pre-trained data.
""")