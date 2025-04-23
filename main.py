import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageFile
import time
import io

# Enable loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

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

def toggle_camera():
    st.session_state['camera_enabled'] = not st.session_state['camera_enabled']
    st.session_state['frame_count'] = 0

# Safe image display function
def safe_image_display(image, caption):
    try:
        # Removed the use_container_width parameter
        st.image(image, caption=caption)
    except Exception as e:
        st.error(f"Error displaying image: {e}")

if detection_mode == "Upload Image":
    st.header("Upload an Image for Face Detection")
    
    # Add some sliders for face detection parameters
    st.sidebar.subheader("Detection Parameters")
    scale_factor = st.sidebar.slider("Scale Factor", 1.05, 1.5, 1.1, 0.05)
    min_neighbors = st.sidebar.slider("Min Neighbors", 1, 10, 5, 1)
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            # Read the image bytes
            image_bytes = uploaded_file.read()
            
            # Create a BytesIO object from the image bytes
            image_bytes_io = io.BytesIO(image_bytes)
            
            # Open the image using PIL
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
    st.header("Real-time Face Detection using Webcam")
    
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
                        # Removed use_container_width parameter
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