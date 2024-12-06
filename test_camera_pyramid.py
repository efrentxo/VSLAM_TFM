import numpy as np
import matplotlib.pyplot as plt


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


# Example: Define camera poses
R1 = np.eye(3)  # Camera 1 at origin
t1 = np.zeros((3, 1))

R2 = np.array([[0.866, -0.5, 0],  # Camera 2 rotated 30 degrees around Z-axis
               [0.5, 0.866, 0],
               [0, 0, 1]])
t2 = np.array([[1], [1], [0]])  # Camera 2 shifted in the XY plane

# Plot the cameras as pyramids
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

plot_camera_pyramid(ax, R1, t1, scale=0.5, color='cyan', label='Camera 1')
plot_camera_pyramid(ax, R2, t2, scale=0.5, color='magenta', label='Camera 2')

# Set axis labels and limits
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-1, 2)
plt.legend()
plt.title("Camera Poses as 3D Pyramids")
plt.show()