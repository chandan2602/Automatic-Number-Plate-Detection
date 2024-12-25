import cv2
import numpy as np
import imutils
import streamlit as st
from PIL import Image

# Function to detect license plate
def detect_license_plate(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Canny Edge Detection
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours in the edged image
    contours = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    # Sort contours by area, keeping only the largest ones
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    # Initialize license plate contour
    license_plate_contour = None

    # Loop through contours to find a rectangular region
    for contour in contours:
        # Approximate the contour
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # Look for a rectangle (4 sides)
        if len(approx) == 4:
            license_plate_contour = approx
            break

    if license_plate_contour is not None:
        # Create a mask for the detected license plate
        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [license_plate_contour], -1, 255, -1)

        # Crop the bounding rectangle of the license plate
        x, y, w, h = cv2.boundingRect(license_plate_contour)
        cropped_license_plate = image[y:y + h, x:x + w]

        return cropped_license_plate

    return None

# Streamlit App
st.set_page_config(page_title="Number Plate Detection", layout="wide")
st.markdown(
    """
    <style>
    /* Styling the page with modern, Google-like elements */
    body {
        font-family: 'Roboto', sans-serif;
        background-color: #F1F3F4;
    }
    .title {
        text-align: center;
        font-size: 42px;
        font-weight: 600;
        color: #1a73e8;
        margin-top: 20px;
    }
    .subheader {
        text-align: center;
        font-size: 18px;
        color: #555;
        margin-bottom: 20px;
    }
    .card {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        padding: 20px;
        margin-top: 20px;
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 20px;
    }
    .image-container {
        width: 100%;
        height: auto;
        border-radius: 10px;
        overflow: hidden;
        border: 2px solid #4CAF50;
    }
    .upload-container {
        text-align: center;
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .error {
        color: red;
        font-weight: bold;
        text-align: center;
    }
    .btn {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        font-size: 16px;
        cursor: pointer;
    }
    .btn:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="title">📸 Automatic Number Plate Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Upload a car image to detect the number plate!</div>', unsafe_allow_html=True)

# Upload image
uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Load the image
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Resize for display
    image = imutils.resize(image, width=600)

    # Detect the license plate
    cropped_license_plate = detect_license_plate(image)

    if cropped_license_plate is not None:
        # Convert images to RGB for display
        original_image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cropped_license_plate_rgb = cv2.cvtColor(cropped_license_plate, cv2.COLOR_BGR2RGB)

        # Display images in a stylish card layout
        col1, col2 = st.columns(2)

        with col1:
            st.image(original_image_rgb, caption="Original Image", use_container_width=True)

        with col2:
            st.image(cropped_license_plate_rgb, caption="Detected Number Plate", use_container_width=True)
    else:
        st.markdown('<div class="error">No number plate detected. Try another image!</div>', unsafe_allow_html=True)
