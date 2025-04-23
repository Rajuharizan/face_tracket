import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageFile
import face_recognition
import time
import io
import os
from datetime import datetime

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
if 'known_face_encodings' not in st.session_state:
    st.session_state['known_face_encodings'] = []
if 'known_face_names' not in st.session_state:
    st.session_state['known_face_names'] = []
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
        # Create a copy of the image to avoid modifying the original
        image_copy = image.copy()
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
        
        # Check if face_cascade is properly loaded
        if face_cascade is None:
            return image_copy, 0
            
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size
        )
        
        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(image_copy, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        return image_copy, len(faces)
    except Exception as e:
        st.error(f"Error during face detection: {e}")
        return image, 0

def recognize_faces(image):
    """Recognize faces in an image and label them with known names."""
    try:
        # Convert BGR to RGB for face_recognition
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Find all face locations and encodings
        face_locations = face_recognition.face_locations(rgb_image)
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        
        # Copy the image for drawing
        image_copy = image.copy()
        
        # Loop through each face found in the image
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # See if the face is a match for known faces
            matches = face_recognition.compare_faces(st.session_state['known_face_encodings'], face_encoding, tolerance=0.6)
            name = "Unknown"
            
            # If there's a match, use the first one
            if True in matches:
                first_match_index = matches.index(True)
                name = st.session_state['known_face_names'][first_match_index]
            
            # Draw a rectangle around the face
            cv2.rectangle(image_copy, (left, top), (right, bottom), (0, 0, 255), 2)
            
            # Draw a label with a name below the face
            cv2.rectangle(image_copy, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(image_copy, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)
        
        return image_copy, len(face_locations)
    except Exception as e:
        st.error(f"Error during face recognition: {e}")
        return image, 0

def register_face(image, name):
    """Register a new face with a name."""
    try:
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_image)
        
        if len(face_locations) == 0:
            return False, "No faces detected in the image."
        
        if len(face_locations) > 1:
            return False, "Multiple faces detected. Please use an image with only one face."
        
        # Get the encoding of the face
        face_encoding = face_recognition.face_encodings(rgb_image, face_locations)[0]
        
        # Check if this face is already registered
        if st.session_state['known_face_encodings']:
            matches = face_recognition.compare_faces(st.session_state['known_face_encodings'], face_encoding)
            if True in matches:
                return False, f"This face is already registered as {st.session_state['known_face_names'][matches.index(True)]}"
        
        # Add the face encoding and name to the lists
        st.session_state['known_face_encodings'].append(face_encoding)
        st.session_state['known_face_names'].append(name)
        
        # Save the face image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.jpg"
        filepath = os.path.join(st.session_state['registered_faces_dir'], filename)
        
        # Convert back to BGR for saving
        cv2.imwrite(filepath, image)
        
        return True, f"Face registered successfully as {name}!"
    except Exception as e:
        return False, f"Error registering face: {e}"

def load_registered_faces():
    """Load registered faces from the directory."""
    try:
        # Clear existing data
        st.session_state['known_face_encodings'] = []
        st.session_state['known_face_names'] = []
        
        # Check if directory exists
        if not os.path.exists(st.session_state['registered_faces_dir']):
            return 0
        
        count = 0
        # Look for image files in the directory
        for filename in os.listdir(st.session_state['registered_faces_dir']):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                # Extract the name from the filename (format: name_timestamp.jpg)
                name = filename.split('_')[0]
                
                # Load the image
                image_path = os.path.join(st.session_state['registered_faces_dir'], filename)
                image = face_recognition.load_image_file(image_path)
                
                # Get face encodings
                face_encodings = face_recognition.face_encodings(image)
                
                if face_encodings:
                    st.session_state['known_face_encodings'].append(face_encodings[0])
                    st.session_state['known_face_names'].append(name)
                    count += 1
        
        return count
    except Exception as e:
        st.error(f"Error loading registered faces: {e}")
        return 0

def toggle_camera():
    st.session_state['camera_enabled'] = not st.session_state['camera_enabled']
    st.session_state['frame_count'] = 0

# Safe image display function
def safe_image_display(image, caption):
    try:
        st.image(image, caption=caption)
    except Exception as e:
        st.error(f"Error displaying image: {e}")

# Load registered faces when the app starts
num_faces = load_registered_faces()
st.sidebar.info(f"Loaded {num_faces} registered faces")

# Registration tab in sidebar
with st.sidebar.expander("Register New Face"):
    reg_name = st.text_input("Enter name")
    reg_file = st.file_uploader("Upload face image", type=["jpg", "jpeg", "png"])
    register_btn = st.button("Register Face")
    
    if register_btn and reg_name and reg_file:
        try:
            # Read the image bytes
            image_bytes = reg_file.read()
            image_bytes_io = io.BytesIO(image_bytes)
            image = Image.open(image_bytes_io)
            
            # Convert PIL Image to OpenCV format
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Register the face
            success, message = register_face(image_cv, reg_name)
            if success:
                st.sidebar.success(message)
            else:
                st.sidebar.error(message)
        except Exception as e:
            st.sidebar.error(f"Error registering face: {e}")

# Main application logic
if mode == "Face Detection":
    st.header("Face Detection")
    
    if detection_mode == "Upload Image":
        # Add some sliders for face detection parameters
        st.sidebar.subheader("Detection Parameters")
        scale_factor = st.sidebar.slider("Scale Factor", 1.05, 1.5, 1.1, 0.05)
        min_neighbors = st.sidebar.slider("Min Neighbors", 1, 10, 5, 1)
        
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            try:
                # Read the image bytes
                image_bytes = uploaded_file.read()
                image_bytes_io = io.BytesIO(image_bytes)
                image = Image.open(image_bytes_io)
                
                # Display the original image
                safe_image_display(image, "Uploaded Image")
                
                # Convert PIL Image to OpenCV format
                image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                # Process the image with error handling
                processed_image, num_faces = detect_faces(
                    image_cv, 
                    scale_factor=scale_factor,
                    min_neighbors=min_neighbors
                )
                
                # Convert back to RGB for display
                processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                
                # Display the processed image
                safe_image_display(processed_image_rgb, f"Detected {num_faces} face(s)")
                
            except Exception as e:
                st.error(f"Error processing image: {e}")
                st.write("Please try a different image or check if the file is corrupted.")
    
    else:  # Webcam mode
        st.header("Live Face Detection")
        
        # Camera control button
        st.button("Start/Stop Webcam", on_click=toggle_camera)
        
        # Frame placeholder
        frame_placeholder = st.empty()
        status_text = st.empty()
        
        # Webcam handling
        if st.session_state['camera_enabled']:
            status_text.text("Starting camera... Please allow access if prompted.")
            try:
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
                        try:
                            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                            frame_placeholder.image(
                                frame_rgb,
                                caption=f"Faces detected: {num_faces} (Frame {st.session_state['frame_count']})"
                            )
                        except Exception as e:
                            status_text.error(f"Error displaying frame: {e}")
                    else:
                        status_text.error("Failed to capture frame")
                    
                    camera.release()
                    
                    # Auto-rerun to create animation effect
                    if st.session_state['camera_enabled']:
                        time.sleep(0.1)  # Small delay to control frame rate
                        st.experimental_rerun()
            except Exception as e:
                status_text.error(f"Camera error: {e}")
                st.session_state['camera_enabled'] = False
        else:
            status_text.text("Webcam is stopped. Click 'Start/Stop Webcam' to begin.")

else:  # Face Recognition mode
    st.header("Face Recognition")
    
    if detection_mode == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            try:
                # Read the image bytes
                image_bytes = uploaded_file.read()
                image_bytes_io = io.BytesIO(image_bytes)
                image = Image.open(image_bytes_io)
                
                # Display the original image
                safe_image_display(image, "Uploaded Image")
                
                # Convert PIL Image to OpenCV format
                image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                # Process the image with error handling
                processed_image, num_faces = recognize_faces(image_cv)
                
                # Convert back to RGB for display
                processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                
                # Display the processed image
                safe_image_display(processed_image_rgb, f"Recognized {num_faces} face(s)")
                
            except Exception as e:
                st.error(f"Error processing image: {e}")
                st.write("Please try a different image or check if the file is corrupted.")
    
    else:  # Webcam mode
        st.header("Live Face Recognition")
        
        # Camera control button
        st.button("Start/Stop Webcam", on_click=toggle_camera)
        
        # Frame placeholder
        frame_placeholder = st.empty()
        status_text = st.empty()
        
        # Webcam handling
        if st.session_state['camera_enabled']:
            status_text.text("Starting camera... Please allow access if prompted.")
            try:
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
                        processed_frame, num_faces = recognize_faces(frame)
                        try:
                            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                            frame_placeholder.image(
                                frame_rgb,
                                caption=f"Faces recognized: {num_faces} (Frame {st.session_state['frame_count']})"
                            )
                        except Exception as e:
                            status_text.error(f"Error displaying frame: {e}")
                    else:
                        status_text.error("Failed to capture frame")
                    
                    camera.release()
                    
                    # Auto-rerun to create animation effect
                    if st.session_state['camera_enabled']:
                        time.sleep(0.1)  # Small delay to control frame rate
                        st.experimental_rerun()
            except Exception as e:
                status_text.error(f"Camera error: {e}")
                st.session_state['camera_enabled'] = False
        else:
            status_text.text("Webcam is stopped. Click 'Start/Stop Webcam' to begin.")

# Instructions
st.sidebar.header("Instructions")
st.sidebar.info("""
1. Select Mode: Choose between Face Detection or Face Recognition
2. Select Input Mode: Upload Image or Use Webcam
3. Register faces (for recognition) using the sidebar form
4. For webcam: Click the Start/Stop button
""")

# Add information about the application
st.sidebar.header("About")
st.sidebar.info("""
This application uses OpenCV for face detection and the face_recognition library for face recognition.

- Face Detection: Identifies faces in images or video
- Face Recognition: Identifies and names known faces

To use recognition, you must first register faces using the sidebar form.
""")