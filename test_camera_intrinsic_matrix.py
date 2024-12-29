import numpy as np
import cv2
def calculate_intrinsic_matrix(focal_length, skew, principal_point):
    """
    Calculates the camera intrinsic matrix.

    :param focal_length: Tuple (fx, fy) representing the focal lengths in pixels.
    :param skew: Skew coefficient (usually 0 for most cameras).
    :param principal_point: Tuple (cx, cy) for the image center in pixels.
    :return: 3x3 intrinsic matrix.
    """
    fx, fy = focal_length
    cx, cy = principal_point
    K = np.array([
        [fx, skew, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    return K

# Load the image
image = cv2.imread('imagen1.jpg')

# Get dimensions
height, width, channels = image.shape  # Channels will be 3 for RGB and 1 for grayscale

# Calculate total pixels
total_pixels = height * width

print(f"Width: {width}, Height: {height}, Channels: {channels}")
print(f"Total pixels: {total_pixels}")

# Example values
focal_length = (800, 800)  # Example focal lengths in pixels
skew = 0  # Most cameras have zero skew
principal_point = (width/2, height/2)  # Example image center for a 1280x720 image

intrinsic_matrix = calculate_intrinsic_matrix(focal_length, skew, principal_point)
print("Camera Intrinsic Matrix:")
print(intrinsic_matrix)
