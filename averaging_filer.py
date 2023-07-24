
import cv2
import numpy as np

# Define the averaging filter function
def averaging_filter(image, N):
    # Define the filter kernel of size NxN
    kernel = np.ones((N, N)) / (N * N)

    # Convolve the kernel with the input image using a nested loop
    output_image = np.zeros_like(image)
    for i in range(N // 2, image.shape[0] - N // 2):
        for j in range(N // 2, image.shape[1] - N // 2):
            window = image[i - N // 2:i + N // 2 + 1, j - N // 2:j + N // 2 + 1]
            output_image[i, j] = int(round(np.sum(kernel * window)))

    return output_image

# Load the input image as a grayscale image
image = cv2.imread('image_2.jpg', cv2.IMREAD_GRAYSCALE)

# Apply the averaging filter using the defined function
output_image = averaging_filter(image, N=3)

# Display the original and filtered images
cv2.imshow('Original Image', image)
cv2.imshow('Filtered Image', output_image)
cv2.waitKey(0)

# Load a different input image
image = cv2.imread('image_1.jpg', cv2.IMREAD_GRAYSCALE)

# Apply the averaging filter using the defined function
output_image = averaging_filter(image, N=5)

# Display the original and filtered images
cv2.imshow('Original Image', image)
cv2.imshow('Filtered Image', output_image)
cv2.waitKey(0)
