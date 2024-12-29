import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def calculate_sfm(frame1, frame2, K):
    """
    Perform Structure from Motion (SfM) with two frames.

    Args:
    - frame1: First image (grayscale).
    - frame2: Second image (grayscale).
    - K: Camera intrinsic matrix (3x3).

    Returns:
    - points_3d: Reconstructed 3D points (Nx3).
    - R: Relative rotation matrix (3x3).
    - t: Relative translation vector (3x1).
    """
    # Detect and match features
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(frame1, None)
    kp2, des2 = orb.detectAndCompute(frame2, None)
    print("Número de keypoints detectados en imagen 1: ", len(kp1))
    print("Número de keypoints detectados en imagen 2: ", len(kp1))

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    print("Número de keypoints detectados en imagen 1 despues del match: ", len(pts1))
    print("Número de keypoints detectados en imagen 2 despues del match: ", len(pts2))

    # Compute Essential Matrix
    # E, _ = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

    # Recover Pose
    # _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)

    # Filter points using the mask
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]
    pts1_filt = pts1   # pts1[mask.ravel() == 1]
    pts2_filt = pts2   # pts2[mask.ravel() == 1]
    print("Número de keypoints detectados en imagen 1 despues de RANSAC: ", len(pts1_filt))
    print("Número de keypoints detectados en imagen 2 despues de RANSAC: ", len(pts2_filt))

    # Recover Pose using points after RANSAC
    # _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
    _, R_filt, t_filt, _ = cv2.recoverPose(E, pts1_filt, pts2_filt, K)
    matchesMask = mask.ravel().tolist()
    inlier_matches = [m for i, m in enumerate(matches) if matchesMask[i] == 1]

    # Crear plots
    # Draw keypoints on the images
    img1_kp = cv2.drawKeypoints(frame1, kp1, None, color=(0, 255, 0), flags=0)
    img2_kp = cv2.drawKeypoints(frame2, kp2, None, color=(0, 255, 0), flags=0)

    # Draw image with matches before/after RANSAC
    img_matches_all = cv2.drawMatches(frame1, kp1, frame2, kp2, matches, None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    img_matches_inliers = cv2.drawMatches(frame1, kp1, frame2, kp2, inlier_matches, None,
                                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Mostrar imagenes
    if 1 == 1:
        # Display the keypoint image 1
        plt.figure(figsize=(10, 6))
        plt.imshow(img1_kp, cmap='gray')
        plt.title("ORB Keypoints - Image 1 - Keypoints: " + str(len(kp1)))
        plt.axis('off')
        plt.show()

        # Display the keypoint image 2
        plt.figure(figsize=(10, 6))
        plt.imshow(img2_kp, cmap='gray')
        plt.title("ORB  Keypoints - Image 2 - Keypoints: " + str(len(kp2)))
        plt.axis('off')
        plt.show()

        # Plot results
        plt.figure(figsize=(20, 10))
        plt.subplot(2, 1, 1)
        plt.imshow(img_matches_all)
        plt.title("All Matches - Keypoints: " + str(len(matches)))
        plt.axis('off')

        plt.subplot(2, 1, 2)
        plt.imshow(img_matches_inliers)
        plt.title("Inlier Matches (RANSAC) - Keypoints: " + str(len(inlier_matches)))
        plt.axis('off')

        plt.show()

    # Triangulate points
    proj_mat1 = np.hstack((np.eye(3), np.zeros((3, 1))))  # First camera projection matrix
    proj_mat2 = np.hstack((R, t))  # Second camera projection matrix
    proj_mat1 = K @ proj_mat1
    proj_mat2 = K @ proj_mat2
    points_4d_hom = cv2.triangulatePoints(proj_mat1, proj_mat2, pts1.T, pts2.T)
    points_3d = points_4d_hom[:3] / points_4d_hom[3]  # Convert from homogeneous to 3D

    return points_3d.T, R, t


def draw_camera_pyramid(ax, R, t, K, scale=0.5, color='blue', label = 'camara'):
    """
    Draw a camera pose as a pyramid in 3D.

    Args:
    - ax: Matplotlib 3D axis.
    - R: Rotation matrix (3x3).
    - t: Translation vector (3x1).
    - K: Camera intrinsic matrix (3x3).
    - scale: Scaling factor for the pyramid.
    - color: Color of the pyramid.
    """
    # Define pyramid points in camera coordinates
    pyramid_points = np.array([
        [0, 0, 0],  # Camera center
        [-1, -1, 2],  # Bottom-left
        [1, -1, 2],  # Bottom-right
        [1, 1, 2],  # Top-right
        [-1, 1, 2]  # Top-left
    ]) * scale

    # Transform points to world coordinates
    pyramid_points = (R @ pyramid_points.T + t).T

    #test camera
    # Camera location
    # C = R.T + t  # Camera center in world coordinates
    #
    # Plot camera center
    # ax.scatter(C[0], C[1], C[2], c=color, marker='o', label=label)
    #end test camara

    # Draw lines for the pyramid
    base_edges = [[1, 2], [2, 3], [3, 4], [4, 1]]  # Base edges
    for i, j in base_edges:
        ax.plot(
            [pyramid_points[i, 0], pyramid_points[j, 0]],
            [pyramid_points[i, 1], pyramid_points[j, 1]],
            [pyramid_points[i, 2], pyramid_points[j, 2]],
            color=color
        )
    for i in range(1, 5):  # Lines connecting apex to base
        ax.plot(
            [pyramid_points[0, 0], pyramid_points[i, 0]],
            [pyramid_points[0, 1], pyramid_points[i, 1]],
            [pyramid_points[0, 2], pyramid_points[i, 2]],
            color=color
        )


def plot_3d_points_and_cameras(points_3d, R, t, K):
    """
    Plot 3D points and camera poses as pyramids.

    Args:
    - points_3d: Reconstructed 3D points (Nx3).
    - R: Relative rotation matrix (3x3).
    - t: Relative translation vector (3x1).
    - K: Camera intrinsic matrix (3x3).
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot 3D points
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='blue', s=1, label='3D Points')

    # Draw first camera (at origin)
    draw_camera_pyramid(ax, np.eye(3), np.zeros((3, 1)), K, color='cyan', label='Camera 1')

    # Draw second camera
    draw_camera_pyramid(ax, R, t, K, color='red', label='Camera 2')

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title("3D Reconstruction with Camera Poses")
    plt.show()


# Example Usage
if __name__ == "__main__":
    # Load two frames (grayscale)
    frame1 = cv2.imread("imagen1.jpg", cv2.IMREAD_GRAYSCALE)
    frame2 = cv2.imread("imagen2.jpg", cv2.IMREAD_GRAYSCALE)

    # Define the camera intrinsic matrix (example values)
    K = np.array([
        [700, 0, 640],  # Focal length (fx, fy) and principal point (cx, cy)
        [0, 700, 360],
        [0, 0, 1]
    ])

    # Calculate SfM
    points_3d, R, t = calculate_sfm(frame1, frame2, K)

    # Plot results
    plot_3d_points_and_cameras(points_3d, R, t, K)
