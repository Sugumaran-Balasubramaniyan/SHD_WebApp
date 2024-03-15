import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import subprocess
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from heapq import *
from skimage import io
import json







# Function to process the image
def process_image(image, blur_kernel, threshold_value, remove_noise_kernel,
                  erode_vertical_iterations, erode_horizontal_iterations):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply blur
    blurred = cv2.blur(gray, (blur_kernel, blur_kernel))

    # Apply threshold
    thresh = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


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




def crop_rows(processed_image, filename_without_extension ):
    
    output_directory = "output_directory"

    if processed_image.ndim > 2: # is this is a rgb/rgba image
        img = rgb2gray(processed_image)

    def horizontal_projections(sobel_image):
        return np.sum(sobel_image, axis=1)  

    def binarize_image(image):
        threshold = threshold_otsu(img)
        return image < threshold
        
    binarized_image = binarize_image(img)
    hpp = horizontal_projections(binarized_image)

    # find the midway where we can make a threshold and extract the peaks regions
    def find_peak_regions(hpp, threshold):
        
        peaks = []
        for i, hppv in enumerate(hpp):
            if hppv < threshold:
                peaks.append([i, hppv])
        return peaks

    # find the threshold from where anything above is considered a peak region
    # using the average for now but this needs further research. This may or may not work on all images.
    threshold = (np.max(hpp)-np.min(hpp))/2
    peaks = find_peak_regions(hpp, threshold)

    peaks_indexes = np.array(peaks)[:, 0].astype(int)

    segmented_img = np.copy(img)
    r, c = segmented_img.shape
    for ri in range(r):
        if ri in peaks_indexes:
            segmented_img[ri, :] = 0

    # group the peaks through which we will be doing path planning.
    diff_between_consec_numbers = np.diff(peaks_indexes) # difference between consecutive numbers
    indexes_with_larger_diff = np.where(diff_between_consec_numbers > 1)[0].flatten()
    peak_groups = np.split(peaks_indexes, indexes_with_larger_diff)
    # remove very small regions, these are basically errors in algorithm because of our threshold value
    peak_groups = [item for item in peak_groups if len(item) > 10]


    #a star path planning algorithm 
    def heuristic(a, b):
        return (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2

    def astar(array, start, goal):

        neighbors = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
        close_set = set()
        came_from = {}
        gscore = {start:0}
        fscore = {start:heuristic(start, goal)}
        oheap = []

        heappush(oheap, (fscore[start], start))
        
        while oheap:

            current = heappop(oheap)[1]

            if current == goal:
                data = []
                while current in came_from:
                    data.append(current)
                    current = came_from[current]
                return data

            close_set.add(current)
            for i, j in neighbors:
                neighbor = current[0] + i, current[1] + j            
                tentative_g_score = gscore[current] + heuristic(current, neighbor)
                if 0 <= neighbor[0] < array.shape[0]:
                    if 0 <= neighbor[1] < array.shape[1]:                
                        if array[neighbor[0]][neighbor[1]] == 1:
                            continue
                    else:
                        # array bound y walls
                        continue
                else:
                    # array bound x walls
                    continue
                    
                if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                    continue
                    
                if  tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1]for i in oheap]:
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heappush(oheap, (fscore[neighbor], neighbor))
                    
        return []
    
    print("Peak groups found")
    
    # now that everything is cleaner, its time to segment all the lines using the A* algorithm
    def get_binary(img):
        mean = np.mean(img)
        if mean == 0.0 or mean == 1.0:
            return img

        thresh = threshold_otsu(img)
        binary = img <= thresh
        binary = binary * 1
        return binary

    binary_image = get_binary(img)
    segment_separating_lines = []
    
    for i, sub_image_index in enumerate(peak_groups):
        nmap = binary_image[sub_image_index[0]:sub_image_index[-1]]
        path = np.array(astar(nmap, (int(nmap.shape[0]/2), 0), (int(nmap.shape[0]/2),nmap.shape[1]-1)))
        offset_from_top = sub_image_index[0]
        path[:,0] += offset_from_top
        segment_separating_lines.append(path)

    cluster_of_interest = peak_groups[1]

    offset_from_top = cluster_of_interest[0]

    nmap = binary_image[cluster_of_interest[0]:cluster_of_interest[-1],:]
    
    path = np.array(astar(nmap, (int(nmap.shape[0]/2), 0), (int(nmap.shape[0]/2),nmap.shape[1]-1)))
    
    offset_from_top = cluster_of_interest[0]

    
    print("splitting")
    
    # splitting algorithm

    seperated_images = []
    for index, line_segments in enumerate(segment_separating_lines):
        if index < len(segment_separating_lines)-1:
            lower_line = np.min(segment_separating_lines[index][:,0])
            upper_line = np.max(segment_separating_lines[index+1][:,0])
            seperated_images.append(processed_image[lower_line:upper_line])

    # Iterate over the separated images and save them with unique names
    for index, line_image in enumerate(seperated_images):
        save_filename = os.path.join(output_directory, 
                                     f"{os.path.splitext(filename_without_extension)[0]}_line_{index+1}.png")
        cv2.imwrite(save_filename, line_image)


    # Print a message indicating the successful saving of images
    print(f"{len(seperated_images)} images saved.")














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
        threshold_value = st.sidebar.slider("Threshold Value", min_value=0, max_value=255, value=1)
        
        
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



         # Button to crop rows
        if st.sidebar.button("Crop Rows"):
            zoned_image = crop_rows(processed_image, filename_without_extension)



if __name__ == "__main__":
    main()
