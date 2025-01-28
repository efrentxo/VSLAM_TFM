import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def extract_features(frame1, frame2, K, max_features = 500):
    """
    Extraer features de dos imagenes y hacer el match

    Args:
    - frame1: First image (grayscale).
    - frame2: Second image (grayscale).
    - K: Camera intrinsic matrix (3x3).

    Returns:
    - pts1
    - pts2

    """
    # Create an ORB detector with a specific maximum number of features
    orb = cv2.ORB_create(nfeatures=max_features)

    # Detect and match features
    kp1, des1 = orb.detectAndCompute(frame1, None)
    kp2, des2 = orb.detectAndCompute(frame2, None)
    print("Número de keypoints detectados en imagen 1: ", len(kp1))
    print("Número de keypoints detectados en imagen 2: ", len(kp2))

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    return pts1, pts2, kp1, kp2, matches

def recover_camera_pose(E, points1, points2, K):
    """
    Recovers the relative camera pose (R, t) from the essential matrix.

    Parameters:
        E (numpy.ndarray): Essential matrix (3x3).
        points1 (numpy.ndarray): Nx2 array of points in the first image.
        points2 (numpy.ndarray): Nx2 array of points in the second image.
        K (numpy.ndarray): Intrinsic camera matrix (3x3).

    Returns:
        R (numpy.ndarray): Rotation matrix (3x3).
        t (numpy.ndarray): Translation vector (3x1).
        mask (numpy.ndarray): Inlier mask from cv2.recoverPose.
    """
    # Ensure points are in the correct shape (Nx2 to Nx1x2)
    if points1.ndim == 2:
        points1 = points1[:, np.newaxis, :]
    if points2.ndim == 2:
        points2 = points2[:, np.newaxis, :]

    # Recover the pose (R and t)
    retval, R, t, mask = cv2.recoverPose(E, points1, points2, K)

    return retval, R, t, mask

def filter_points_ransac(points1, points2, K, method='essential', reproj_thresh=1.0):
    """
    Filters matching points between two images using RANSAC.

    Parameters:
        points1 (numpy.ndarray): Nx2 array of points in the first image.
        points2 (numpy.ndarray): Nx2 array of points in the second image.
        K:intrinsic matrix
        method (str): The transformation model to use ('fundamental', 'essential', or 'homography').
        reproj_thresh (float): RANSAC reprojection threshold.

    Returns:
        inlier_points1 (numpy.ndarray): Filtered points in the first image.
        inlier_points2 (numpy.ndarray): Filtered points in the second image.
        mask (numpy.ndarray): Boolean mask indicating inlier points.
        model (numpy.ndarray): Estimated transformation matrix (F, E, or H).
    """
    if method == 'fundamental':
        # Compute the fundamental matrix
        model, mask = cv2.findFundamentalMat(points1, points2, K, cv2.FM_RANSAC, reproj_thresh)
    elif method == 'essential':
        # Compute the essential matrix
        model, mask = cv2.findEssentialMat(points1, points2, K, method=cv2.RANSAC, prob=0.999, threshold=reproj_thresh)
    elif method == 'homography':
        # Compute the homography matrix
        model, mask = cv2.findHomography(points1, points2, K, cv2.RANSAC, reproj_thresh)
    else:
        raise ValueError("Unsupported method. Use 'fundamental', 'essential', or 'homography'.")

    # Convert mask to boolean array
    mask = mask.ravel().astype(bool)

    # Filter points using the mask
    inlier_points1 = points1[mask]
    inlier_points2 = points2[mask]

    return inlier_points1, inlier_points2, mask, model

def Extraer3D_points(R_prev, t_prev, R, t, pts1, pts2, K):
    proj_mat1 = np.hstack((R_prev, t_prev))  # First camera projection matrix
    proj_mat2 = np.hstack((R, t))            # Second camera projection matrix
    proj_mat1 = K @ proj_mat1
    proj_mat2 = K @ proj_mat2
    points_4d_hom = cv2.triangulatePoints(proj_mat1, proj_mat2, pts1.T, pts2.T)

    # Convert from homogeneous to 3D
    points_3d = points_4d_hom[:3] / points_4d_hom[3]

    return points_3d.T

def plot_3d_points_and_cameras(points_3d, R, t, K):
    """
    Plot 3D points and camera poses as pyramids.

    Args:
    - points_3d: Reconstructed 3D points (Nx3).
    - R: Relative rotation matrix (3x3).
    - t: Relative translation vector (3x1).
    - K: Camera intrinsic matrix (3x3).
    """
    fig = plt.figure(4)
    ax = fig.add_subplot(111, projection='3d')

    # Plot 3D points
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='blue', s=1, label='3D Points')
    ax.scatter(points_3d[0, 0], points_3d[0, 1], points_3d[0, 2], c='red', marker='*', s=20, label='3D Points')

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
    # Save the figure to a file
    plt.savefig("SLAM_3DView.png")
    plt.show(block=False)

def draw_camera_pyramid(ax, R, t, K, scale=0.1, color='blue', label = 'camara'):
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
    pyramid_points_test = (R @ pyramid_points.T + t).T

    #test camera
    # Camera location
    # C = R.T + t  # Camera center in world coordinates
    #
    # Plot camera center
    ax.scatter(pyramid_points[0,0], pyramid_points[0,1], pyramid_points[0,2], c=color, marker='o', label=label)
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

def plot_3d_points(points_3d):
    """
    Plot 3D points

    Args:
    - points_3d: Reconstructed 3D points (Nx3).

    """
    fig = plt.figure(5)
    ax = fig.add_subplot(111, projection='3d')

    # Plot 3D points
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='blue', s=1, label='3D Points')
    # ax.scatter(points_3d[0, 0], points_3d[0, 1], points_3d[0, 2], c='red', marker='*', s=20, label='3D Points')

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title("3D Reconstruction with Camera Poses")
    # Save the figure to a file
    plt.savefig("SLAM_3DView.png")
    plt.show(block=False)

def calculate_camera_pose(initial_R, initial_t, relative_R, relative_t):
    """
    Calculates the camera pose in the global frame based on the initial pose and relative pose.

    Parameters:
        initial_R (numpy.ndarray): Initial rotation matrix (3x3) of the camera in the global frame.
        initial_t (numpy.ndarray): Initial translation vector (3x1) of the camera in the global frame.
        relative_R (numpy.ndarray): Relative rotation matrix (3x3) of the camera for the current frame.
        relative_t (numpy.ndarray): Relative translation vector (3x1) of the camera for the current frame.

    Returns:
        global_R (numpy.ndarray): Rotation matrix (3x3) of the camera in the global frame.
        global_t (numpy.ndarray): Translation vector (3x1) of the camera in the global frame.
    """
    # Update rotation: Combine global rotation with relative rotation
    global_R = np.dot(initial_R, relative_R)

    # Update translation: Combine global translation with transformed relative translation
    global_t = initial_t + np.dot(initial_R, relative_t)

    return global_R, global_t

def plot_camera_pyramid(ax, R, t, scale=1.0, color='b', label=None):
    """
    Plot a camera as a 3D pyramid in space.

    Args:
    - ax: Matplotlib 3D axis.
    - R: Rotation matrix (3x3).
    - t: Translation vector (3x1).
    - scale: Size of the pyramid.
    - color: Color of the pyramid.
    - label: Label for the camera.
    """
    # Camera location
    C = -R.T @ t  # Camera center in world coordinates

    # Define frustum vertices in the camera frame
    # These represent the field of view
    image_plane_distance = scale
    frustum_vertices_camera = np.array([
        [0, 0, 0],  # Camera center
        [-0.5, -0.5, image_plane_distance],  # Bottom-left of the image plane
        [0.5, -0.5, image_plane_distance],  # Bottom-right of the image plane
        [0.5, 0.5, image_plane_distance],  # Top-right of the image plane
        [-0.5, 0.5, image_plane_distance]  # Top-left of the image plane
    ])

    # Transform vertices to world coordinates
    frustum_vertices_world = (R.T @ frustum_vertices_camera.T).T + C.reshape(1, -1)

    # Plot camera center
    ax.scatter(C[0], C[1], C[2], c=color, marker='o', label=label)

    # Draw edges from the camera center to the image plane
    for i in range(1, 5):
        ax.plot(
            [frustum_vertices_world[0, 0], frustum_vertices_world[i, 0]],
            [frustum_vertices_world[0, 1], frustum_vertices_world[i, 1]],
            [frustum_vertices_world[0, 2], frustum_vertices_world[i, 2]],
            color=color
        )

    # Draw the image plane (base of the pyramid)
    base_indices = [1, 2, 3, 4, 1]
    for i in range(len(base_indices) - 1):
        ax.plot(
            [frustum_vertices_world[base_indices[i], 0], frustum_vertices_world[base_indices[i + 1], 0]],
            [frustum_vertices_world[base_indices[i], 1], frustum_vertices_world[base_indices[i + 1], 1]],
            [frustum_vertices_world[base_indices[i], 2], frustum_vertices_world[base_indices[i + 1], 2]],
            color=color
        )

def plot_camera_triangle(ax, R, t, scale=1.0, color='b', label=None):
    """
    Plot a camera as a triangle in 3D space.

    Args:
    - ax: Matplotlib 3D axis.
    - R: Rotation matrix (3x3).
    - t: Translation vector (3x1).
    - scale: Size of the triangle.
    - color: Color of the triangle.
    - label: Label for the camera.
    """
    # Camera location
    C = -R.T @ t  # Camera center in world coordinates

    # Define triangle vertices in the camera frame
    # These represent points on the image plane
    image_plane_distance = scale
    triangle_vertices_camera = np.array([
        [0, 0, 0],  # Camera center
        [-0.5, -0.5, image_plane_distance],  # Bottom-left of the image plane
        [0.5, -0.5, image_plane_distance],  # Bottom-right of the image plane
        [0, 0.5, image_plane_distance]  # Top of the image plane
    ]).T

    # Transform vertices to world coordinates
    triangle_vertices_world = R.T @ triangle_vertices_camera + C.reshape(-1, 1)

    # Plot camera center
    ax.scatter(C[0], C[1], C[2], c=color, marker='o', label=label)

    # Draw edges of the triangle (lines connecting the vertices)
    for i in range(1, 4):
        ax.plot(
            # [C[0], triangle_vertices_world[0, i]],
            # [C[1], triangle_vertices_world[1, i]],
            # [C[2], triangle_vertices_world[2, i]],
            [triangle_vertices_world[0, 0], triangle_vertices_world[0, i]],
            [triangle_vertices_world[0, 1], triangle_vertices_world[1, i]],
            [triangle_vertices_world[0, 2], triangle_vertices_world[2, i]],
            color=color
        )

    # Draw the base of the triangle
    base_indices = [1, 2, 3, 1]
    for i in range(len(base_indices) - 1):
        ax.plot(
            [triangle_vertices_world[0, base_indices[i]], triangle_vertices_world[0, base_indices[i + 1]]],
            [triangle_vertices_world[1, base_indices[i]], triangle_vertices_world[1, base_indices[i + 1]]],
            [triangle_vertices_world[2, base_indices[i]], triangle_vertices_world[2, base_indices[i + 1]]],
            color=color
        )

def plot_camera_pose(ax, R, t, color='blue', label=None):
    """
    Plot a camera pose in 3D.

    Args:
    - R: Rotation matrix (3x3).
    - t: Translation vector (3x1).
    - ax: Matplotlib 3D axis.
    - color: Color for the camera representation.
    - label: Label for the camera.
    """
    # Camera center in world coordinates
    C = -R.T @ t

    # Plot camera center
    ax.scatter(C[0], C[1], C[2], color=color, label=label)
    ax.quiver(C[0], C[1], C[2], R[0, 0], R[1, 0], R[2, 0], length=0.1, color='red')  # X-axis
    ax.quiver(C[0], C[1], C[2], R[0, 1], R[1, 1], R[2, 1], length=0.1, color='green')  # Y-axis
    ax.quiver(C[0], C[1], C[2], R[0, 2], R[1, 2], R[2, 2], length=0.1, color='blue')  # Z-axis


# Function to update the plot incrementally
def update_plot(new_map_points, new_trajectory_point):
    global map_points_handle, trajectory_handle

    # Update map points
    if new_map_points is not None:
        map_points.extend(new_map_points)
        if map_points_handle is None:  # First time plotting
            map_points_handle = ax.scatter(
                [p[0] for p in map_points],
                [p[1] for p in map_points],
                [p[2] for p in map_points],
                c='b', s=5, label="Map Points"
            )
        else:  # Update existing scatter plot
            map_points_handle._offsets3d = (
                [p[0] for p in map_points],
                [p[1] for p in map_points],
                [p[2] for p in map_points]
            )

    # Update trajectory
    if new_trajectory_point is not None:
        trajectory.append(new_trajectory_point)
        if trajectory_handle is None:  # First time plotting
            trajectory_handle = ax.plot(
                [p[0] for p in trajectory],
                [p[1] for p in trajectory],
                [p[2] for p in trajectory],
                c='r', label="Trajectory"
            )[0]
        else:  # Update existing line plot
            trajectory_handle.set_data(
                [p[0] for p in trajectory],
                [p[1] for p in trajectory]
            )
            trajectory_handle.set_3d_properties([p[2] for p in trajectory])

    # Adjust plot limits and draw
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-10, 10)
    ax.legend()
    plt.pause(0.01)