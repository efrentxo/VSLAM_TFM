import numpy as np
import matplotlib.pyplot as plt


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


# Example: Define camera poses
R1 = np.eye(3)  # Camera 1 at origin
t1 = np.zeros((3, 1))

R2 = np.array([[0.866, -0.5, 0],  # Camera 2 rotated 30 degrees around Z-axis
               [0.5, 0.866, 0],
               [0, 0, 1]])
t2 = np.array([[1], [1], [0]])  # Camera 2 shifted in the XY plane

# Plot cameras
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

plot_camera_triangle(ax, R1, t1, scale=0.5, color='red', label='Camera 1')
# plot_camera_triangle(ax, R2, t2, scale=0.5, color='magenta', label='Camera 2')

# Set axis labels and limits
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-1, 2)
plt.legend()
plt.title("Camera Poses as Triangles")
plt.show()
