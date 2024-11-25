################################################# Assignment 2 ###########################################
# Import necessary libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt
from google.colab.patches import cv2_imshow
from google.colab import files

# Upload the image
print("Upload an image:")
uploaded = files.upload()
image_path = list(uploaded.keys())[0]

# Read the image
# Convert it to grayscale since Harris Corner Detection works on single-channel images
image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Convert the image to float32, as required by the Harris algorithm
gray_image_float = np.float32(gray_image)

# Apply the Harris corner detector
# blockSize: size of the neighborhood considered for corner detection
# ksize: aperture parameter of the Sobel derivative used
# k: Harris detector free parameter in the equation (sensitivity factor, typically 0.04-0.06)
block_size = 2
aperture_size = 3
k = 0.04
harris_response = cv2.cornerHarris(gray_image_float, blockSize=block_size, ksize=aperture_size, k=k)

# Dilate the response to mark the corners more visibly
harris_response_dilated = cv2.dilate(harris_response, None)

# Threshold the response to identify strong corners
threshold = 0.01 * harris_response.max()  # Keep only responses greater than 1% of max response
image_with_corners = image.copy()
image_with_corners[harris_response > threshold] = [0, 0, 255]  # Mark corners in red

# Display the original and corner-detected images
print("Original Image:")
cv2_imshow(image)

print("Harris Corner Detection:")
cv2_imshow(image_with_corners)

# Plot the Harris response map using Matplotlib
plt.figure(figsize=(10, 6))
plt.title("Harris Response Map")
plt.imshow(harris_response, cmap='gray')
plt.colorbar()
plt.show()
