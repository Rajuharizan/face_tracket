import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageFile
import time
import io
import os
from datetime import datetime
from deepface import DeepFace

# Enable loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Title of the application
st.title("Face Detection and Recognition System")

# Sidebar for options
st.sidebar.header("Options")
mode = st.sidebar.radio(
    "Select Mode:",
    ("Face Detection", "Face Recognition")
)

if mode == "Face Recognition":
    detection_mode = st.sidebar.radio(
        "Select Input Mode:",
        ("Upload Image", "Use Webcam")
    )
else:
    detection_mode = st.sidebar.radio(
        "Select Input Mode:",
        ("Upload Image", "Use Webcam")
    )

# Initialize session state variables
if 'camera_enabled' not in st.session_state:
    st.session_state['camera_enabled'] = False
if 'frame_count' not in st.session_state:
    st.session_state['frame_count'] = 0
if 'registered_faces_dir' not in st.session_state:
    st.session_state['registered_faces_dir'] = "registered_faces"

# Create directory for registered faces if it doesn't exist
if not os.path.exists(st.session_state['registered_faces_dir']):
    os.makedirs(st.session_state['registered_faces_dir'])

# Function to safely load the cascade classifier
@st.cache_resource
def load_cascade_classifier():
    try:
        classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if classifier.empty():
            st.error("Error: Haar cascade classifier couldn't be loaded properly")
            return None
        return classifier
    except Exception as e:
        st.error(f"Error loading cascade classifier: {e}")
        return None

face_cascade = load_cascade_classifier()

def detect_faces(image, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
    """Detect faces in an image and draw rectangles around them."""
    try:
        image_copy = image.copy()
        gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
        
        if face_cascade is None:
            return image_copy, 0
            
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size
        )
        
        for (x, y, w, h) in faces:
            cv2.rectangle(image_copy, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        return image_copy, len(faces)
    except Exception as e:
        st.error(f"Error during face detection: {e}")
        return image, 0

def recognize_faces(image):
    """Recognize faces in an image using DeepFace."""
    try:
        image_copy = image.copy()
        rgb_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
        
        # Save temporary image for DeepFace processing
        temp_path = "temp.jpg"
        cv2.imwrite(temp_path, image_copy)
        
        # Find all faces using DeepFace
        try:
            results = DeepFace.find(
                img_path=temp_path,
                db_path=st.session_state['registered_faces_dir'],
                enforce_detection=False,
                silent=True
            )
            
            if not isinstance(results, list):
                results = [results]
                
            face_count = 0
            
            for result in results:
                if not result.empty:
                    face_count += 1
                    identity = os.path.basename(result['identity'].iloc[0]).split('_')[0]
                    x, y, w, h = result['source_x'].iloc[0], result['source_y'].iloc[0], \
                                result['source_w'].iloc[0], result['source_h'].iloc[0]
                    
                    # Draw rectangle and label
                    cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.rectangle(image_copy, (x, y-35), (x+w, y), (0, 0, 255), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(image_copy, identity, (x + 6, y - 6), font, 0.8, (255, 255, 255), 1)
        
        except ValueError as e:
            if "No face detected" in str(e):
                face_count = 0
            else:
                raise e
        
        os.remove(temp_path)
        return image_copy, face_count
    except Exception as e:
        st.error(f"Error during face recognition: {e}")
        return image, 0

def register_face(image, name):
    """Register a new face with a name."""
    try:
        # Save the image in the registered faces directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.jpg"
        filepath = os.path.join(st.session_state['registered_faces_dir'], filename)
        cv2.imwrite(filepath, image)
        return True, f"Face registered successfully as {name}!"
    except Exception as e:
        return False, f"Error registering face: {e}"

def toggle_camera():
    st.session_state['camera_enabled'] = not st.session_state['camera_enabled']
    st.session_state['frame_count'] = 0

def safe_image_display(image, caption):
    try:
        st.image(image, caption=caption)
    except Exception as e:
        st.error(f"Error displaying image: {e}")

# Main application logic
if mode == "Face Detection":
    st.header("Face Detection")
    
    if detection_mode == "Upload Image":
        scale_factor = st.sidebar.slider("Scale Factor", 1.05, 1.5, 1.1, 0.05)
        min_neighbors = st.sidebar.slider("Min Neighbors", 1, 10, 5, 1)
        
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                safe_image_display(image, "Uploaded Image")
                image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                processed_image, num_faces = detect_faces(
                    image_cv, 
                    scale_factor=scale_factor,
                    min_neighbors=min_neighbors
                )
                processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                safe_image_display(processed_image_rgb, f"Detected {num_faces} face(s)")
            except Exception as e:
                st.error(f"Error processing image: {e}")
    
    else:  # Webcam mode
        st.header("Live Face Detection")
        st.button("Start/Stop Webcam", on_click=toggle_camera)
        frame_placeholder = st.empty()
        status_text = st.empty()
        
        if st.session_state['camera_enabled']:
            status_text.text("Starting camera...")
            try:
                camera = cv2.VideoCapture(0)
                
                if not camera.isOpened():
                    status_text.error("Could not access webcam.")
                    st.session_state['camera_enabled'] = False
                else:
                    status_text.text("Camera is running.")
                    ret, frame = camera.read()
                    if ret:
                        st.session_state['frame_count'] += 1
                        processed_frame, num_faces = detect_faces(frame)
                        frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(
                            frame_rgb,
                            caption=f"Faces detected: {num_faces} (Frame {st.session_state['frame_count']})"
                        )
                    camera.release()
                    if st.session_state['camera_enabled']:
                        time.sleep(0.1)
                        st.experimental_rerun()
            except Exception as e:
                status_text.error(f"Camera error: {e}")
                st.session_state['camera_enabled'] = False
        else:
            status_text.text("Webcam is stopped.")

else:  # Face Recognition mode
    st.header("Face Recognition")
    
    if detection_mode == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                safe_image_display(image, "Uploaded Image")
                image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                processed_image, num_faces = recognize_faces(image_cv)
                processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                safe_image_display(processed_image_rgb, f"Recognized {num_faces} face(s)")
            except Exception as e:
                st.error(f"Error processing image: {e}")
    
    else:  # Webcam mode
        st.header("Live Face Recognition")
        st.button("Start/Stop Webcam", on_click=toggle_camera)
        frame_placeholder = st.empty()
        status_text = st.empty()
        
        if st.session_state['camera_enabled']:
            status_text.text("Starting camera...")
            try:
                camera = cv2.VideoCapture(0)
                
                if not camera.isOpened():
                    status_text.error("Could not access webcam.")
                    st.session_state['camera_enabled'] = False
                else:
                    status_text.text("Camera is running.")
                    ret, frame = camera.read()
                    if ret:
                        st.session_state['frame_count'] += 1
                        processed_frame, num_faces = recognize_faces(frame)
                        frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(
                            frame_rgb,
                            caption=f"Faces recognized: {num_faces} (Frame {st.session_state['frame_count']})"
                        )
                    camera.release()
                    if st.session_state['camera_enabled']:
                        time.sleep(0.1)
                        st.experimental_rerun()
            except Exception as e:
                status_text.error(f"Camera error: {e}")
                st.session_state['camera_enabled'] = False
        else:
            status_text.text("Webcam is stopped.")

# Registration tab in sidebar
with st.sidebar.expander("Register New Face"):
    reg_name = st.text_input("Enter name")
    reg_file = st.file_uploader("Upload face image", type=["jpg", "jpeg", "png"])
    register_btn = st.button("Register Face")
    
    if register_btn and reg_name and reg_file:
        try:
            image = Image.open(reg_file)
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            success, message = register_face(image_cv, reg_name)
            if success:
                st.sidebar.success(message)
            else:
                st.sidebar.error(message)
        except Exception as e:
            st.sidebar.error(f"Error registering face: {e}")

# Instructions and About
st.sidebar.header("Instructions")
st.sidebar.info("""
1. Select Mode: Face Detection or Recognition
2. Select Input Mode: Upload Image or Webcam
3. Register faces using the sidebar form
""")

st.sidebar.header("About")
st.sidebar.info("""
This application uses:
- OpenCV for face detection
- DeepFace for face recognition
""")