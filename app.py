import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

# Function to process the image
def process_image(image, blur_kernel, threshold_value, remove_noise_kernel,
                  erode_vertical_iterations, erode_horizontal_iterations):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply blur
    blurred = cv2.blur(gray, (blur_kernel, blur_kernel))

    # Apply threshold
    _, thresh = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)

    # Invert the image
    inverted = cv2.bitwise_not(thresh)

    # Erode vertical lines
    vertical_kernel = np.ones((1, 6), np.uint8)
    vertical_lines_eroded_image = cv2.erode(inverted, vertical_kernel, iterations=erode_vertical_iterations)
    vertical_lines_eroded_image = cv2.dilate(vertical_lines_eroded_image, vertical_kernel, iterations=erode_vertical_iterations)

    # Erode horizontal lines
    horizontal_kernel = np.ones((6, 1), np.uint8)
    horizontal_lines_eroded_image = cv2.erode(inverted, horizontal_kernel, iterations=erode_horizontal_iterations)
    horizontal_lines_eroded_image = cv2.dilate(horizontal_lines_eroded_image, horizontal_kernel, iterations=erode_horizontal_iterations)

    # Combine eroded images
    combined_image = cv2.addWeighted(vertical_lines_eroded_image, 0.5, horizontal_lines_eroded_image, 0.5, 0)

    # Dilate combined image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    combined_image_dilated = cv2.dilate(combined_image, kernel, iterations=5)

    # Subtract lines
    image_without_lines = cv2.subtract(inverted, combined_image_dilated)

    # Apply noise reduction
    denoised = cv2.fastNlMeansDenoising(image_without_lines, h=remove_noise_kernel)

    return denoised



# Streamlit app
def main():
    st.title("Image Processing App")

    # Load the image from file
    image_path = st.sidebar.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    if image_path is not None:
        # Read image using OpenCV
        image = np.array(Image.open(image_path))
        st.image(image, caption="Original Image", use_column_width=True)

        # Sidebar sliders for adjustable variables
        blur_kernel = st.sidebar.slider("Blur Kernel Size", min_value=1, max_value=15, value=5)
        threshold_value = st.sidebar.slider("Threshold Value", min_value=0, max_value=255, value=127)
        
        
        erode_vertical_iterations = st.sidebar.slider("Vertical Line Erosion Iterations", min_value=1, max_value=20, value=10)
        erode_horizontal_iterations = st.sidebar.slider("Horizontal Line Erosion Iterations", min_value=1, max_value=20, value=10)

        remove_noise_kernel = st.sidebar.slider("Remove Noise", min_value=1, max_value=100, value=10)

        # Process the image
        processed_image = process_image(image, blur_kernel, threshold_value, remove_noise_kernel,
                                         erode_vertical_iterations, erode_horizontal_iterations)
        st.image(processed_image, caption="Processed Image", use_column_width=True)

        # Button to save the processed image
        if st.button('Save Processed Image'):
            # Remove the file extension from the filename
            filename = image_path.name
            filename_without_extension = os.path.splitext(filename)[0]

            # Save the processed image with the new filename
            output_filename = filename_without_extension + "_processed.jpg"
            cv2.imwrite(output_filename, processed_image)
            st.success(f"Processed image saved as {output_filename} successfully.")

if __name__ == "__main__":
    main()
