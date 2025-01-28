import cv2
import numpy as np
import matplotlib.pyplot as plt


def calculate_sfm(frame1, frame2, K):
    """
    Calculate Structure from Motion (SfM) using two frames.

    Args:
    - frame1: First image (grayscale).
    - frame2: Second image (grayscale).
    - K: Camera intrinsic matrix (3x3).

    Returns:
    - points_3d: Reconstructed 3D points.
    - R: Relative rotation matrix (3x3).
    - t: Relative translation vector (3x1).
    """
    # Step 1: Detect and match features
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(frame1, None)
    kp2, des2 = orb.detectAndCompute(frame2, None)

    # Match features using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract matched keypoints
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.queryIdx].pt for m in matches])

    # Step 2: Compute the Essential Matrix
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

    # Step 3: Recover Pose
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)

    # Filter points using the mask
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    # Step 4: Triangulate points
    proj_mat1 = np.hstack((np.eye(3), np.zeros((3, 1))))  # First camera projection matrix
    proj_mat2 = np.hstack((R, t))  # Second camera projection matrix
    proj_mat1 = K @ proj_mat1
    proj_mat2 = K @ proj_mat2

    points_4d_hom = cv2.triangulatePoints(proj_mat1, proj_mat2, pts1.T, pts2.T)
    points_3d = points_4d_hom[:3] / points_4d_hom[3]  # Convert from homogeneous to 3D

    return points_3d.T, R, t


def plot_3d_points(points_3d, R, t):
    """
    Plot 3D points and camera poses.

    Args:
    - points_3d: 3D points (Nx3).
    - R: Rotation matrix (3x3).
    - t: Translation vector (3x1).
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot 3D points
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='blue', s=1, label='3D Points')

    # Plot cameras
    ax.scatter(0, 0, 0, c='red', marker='o', label='Camera 1')
    camera2_position = -R.T @ t
    ax.scatter(camera2_position[0], camera2_position[1], camera2_position[2], c='green', marker='o', label='Camera 2')

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title("3D Reconstruction")
    plt.show()


# Example Usage
if __name__ == "__main__":
    # Load two frames (grayscale)
    frame1 = cv2.imread("imagen1.jpg", cv2.IMREAD_GRAYSCALE)
    frame2 = cv2.imread("imagen2.jpg", cv2.IMREAD_GRAYSCALE)

    # Define the camera intrinsic matrix (example values)
    K = np.array([
        [700, 0, 640],  # Focal length (fx, fy) and principal point (cx, cy)
        [0, 700, 480],
        [0, 0, 1]
    ])

    # Calculate SfM
    points_3d, R, t = calculate_sfm(frame1, frame2, K)

    # Plot the results
    plot_3d_points(points_3d, R, t)
