import numpy as np

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