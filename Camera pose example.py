import numpy as np
import cv2  # OpenCV for essential matrix and pose recovery

def generate_synthetic_data():
    # Step 1: Generate random 3D points
    num_points = 100
    points_3d = np.random.uniform(-5, 5, (num_points, 3))

    # Camera intrinsics
    fx, fy = 800, 800
    cx, cy = 400, 400
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])

    # Camera poses
    R1 = np.eye(3)
    t1 = np.array([0, 0, 0])
    R2 = cv2.Rodrigues(np.array([0.1, 0.2, 0.3]))[0]  # Slight rotation
    R2 = R1
    t2 = np.array([1, 0, 0])  # Translation along X and Z

    def project_points(points, K, R, t):
        """Project 3D points into 2D."""
        points_h = np.hstack((points, np.ones((points.shape[0], 1))))  # Homogeneous
        extrinsics = np.hstack((R, t.reshape(-1, 1)))  # Combine R and t
        points_cam = (extrinsics @ points_h.T).T  # Transform to camera frame
        points_2d = (K @ points_cam[:, :3].T).T  # Project to image plane
        points_2d = points_2d[:, :2] / points_2d[:, 2:]  # Normalize by depth
        return points_2d

    points_2d_frame1 = project_points(points_3d, K, R1, t1)
    points_2d_frame2 = project_points(points_3d, K, R2, t2)

    return {
        "points_3d": points_3d,
        "points_2d_frame1": points_2d_frame1,
        "points_2d_frame2": points_2d_frame2,
        "K": K,
        "R2": R2,
        "t2": t2
    }

# Step 2: Essential Matrix and Pose Extraction
def pose_extraction(data):
    points_2d_1 = data["points_2d_frame1"]
    points_2d_2 = data["points_2d_frame2"]
    K = data["K"]

    # Normalize points by intrinsic matrix
    points_2d_1_norm = cv2.undistortPoints(np.expand_dims(points_2d_1, axis=1), K, None)
    points_2d_2_norm = cv2.undistortPoints(np.expand_dims(points_2d_2, axis=1), K, None)

    # Compute Essential Matrix
    E, mask = cv2.findEssentialMat(points_2d_1_norm, points_2d_2_norm, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

    # Recover Pose
    _, R, t, mask_pose = cv2.recoverPose(E, points_2d_1_norm, points_2d_2_norm, K)

    return R, t

# Generate synthetic data
data = generate_synthetic_data()

# Extract pose
R, t = pose_extraction(data)

# Print results
print("Ground Truth Rotation:\n", data["R2"])
print("Recovered Rotation:\n", R)
print("\nGround Truth Translation:\n", data["t2"])
print("Recovered Translation (Normalized):\n", t.flatten())

# Draw first camera (at origin)
draw_camera_pyramid(ax, np.eye(3), np.zeros((3, 1)), K, color='cyan', label='Camera 1')

# Draw second camera
draw_camera_pyramid(ax, R, t, K, color='red', label='Camera 2')
