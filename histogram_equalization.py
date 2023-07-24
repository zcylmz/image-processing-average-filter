

import cv2
import numpy as np

# Load the input image as a grayscale image
image = cv2.imread('image_1.jpg', cv2.IMREAD_GRAYSCALE)

# Calculate the histogram of the input image
histogram = [0] * 256
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        intensity = image[i, j]
        histogram[intensity] += 1

# Calculate the cumulative distribution function (CDF) of the histogram
cdf = [0] * 256
cumulative_sum = 0
for i in range(256):
    cumulative_sum += histogram[i]
    cdf[i] = cumulative_sum

# Scale the CDF to map the intensity levels of the input image to new intensity levels
scale_factor = 255 / (image.shape[0] * image.shape[1])
new_intensity_levels = [0] * 256
for i in range(256):
    new_intensity_levels[i] = int(round(scale_factor * cdf[i]))

# Use the mapped intensity levels to create the equalized image
equalized_image = np.zeros_like(image)
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        intensity = image[i, j]
        equalized_image[i, j] = new_intensity_levels[intensity]

# Display the original and equalized images
cv2.imshow('Original Image', image)
cv2.imshow('Equalized Image', equalized_image)
cv2.waitKey(0)



