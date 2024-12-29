import numpy as np

# Intrinsic parameters
f_x, f_y = 800, 800  # Focal lengths in pixels
c_x, c_y = 640, 360  # Optical center (example for 1280x720 resolution)

# Intrinsic matrix K
K = np.array([
    [f_x, 0, c_x],
    [0, f_y, c_y],
    [0, 0, 1]
])

# Rotation matrix (90-degree rotation around the Z-axis for horizontal camera)
R = np.array([
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, 1]
])

# Translation vector (single camera at origin)
T = np.zeros((3, 1))

# Combine R and T
RT = np.hstack((R, T))

# Compute projection matrix
P = K @ RT

print("Intrinsic Matrix (K):\n", K)
print("\nRotation Matrix (R):\n", R)
print("\nTranslation Vector (T):\n", T)
print("\nProjection Matrix (P):\n", P)
