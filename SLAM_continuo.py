# Lbrerias importadas
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# TODO
# [] añadir formato de la camara desde fichero externo matrix intrinseca

# Mis funciones
from my_functions import extract_features, filter_points_ransac, recover_camera_pose, Extraer3D_points, update_plot, plot_3d_points, plot_3d_points_and_cameras, plot_camera_pyramid, plot_camera_pose, plot_camera_triangle

# Specify the folder containing images
folder_path = "/home/efren/Escritorio/TFM/datos/test4/"

# Get all files in the folder and sort them alphabetically
file_names = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff'))])

# List to store the images
images = []
images_list = []
Init = 1

# Define the camera intrinsic matrix (example values)
K = np.array([
    [700.0, 0., 640.],  # Focal length (fx, fy) and principal point (cx, cy)
    [0.,  700., 360.],
    [0.,    0.,   1.]
])
#Nota: el array de K tiene que ser en formato 0.0, la funcion triangulacion da cosas raras, no lo entiendo

# Initial Camera pose
R0 = np.array([[1, 0, 0],
               [0, 1, 0],
               [0, 0, 1]])
t0 = np.array([[0], [0], [0]])  # Example translation

# Maximo numero de puntos para ORB
max_features = 500

# SLAM data storage
map_points = []  # 3D points
trajectory = []  # Robot position over time
iter = 0

# Plot handles for incremental updates
map_points_handle = None
trajectory_handle = None

# Initialize figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

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
    # ax.set_xlim(-5, 5)
    # ax.set_ylim(-5, 5)
    # ax.set_zlim(-5, 5)

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.pause(0.01)

# Iterate over all files in the folder
for file in file_names:
    ruta_completa = os.path.join(folder_path, file)

    # Verificar si es un archivo y tiene extensión de imagen
    if os.path.isfile(ruta_completa) and file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', 'tiff')):
    # if file_name.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', '.gif', 'tiff')):
        img = cv2.imread(ruta_completa)
        images.append(img)
        images_list.append(ruta_completa)
        frame = cv2.imread(ruta_completa, cv2.IMREAD_GRAYSCALE)

        # Inizializacion con el primer frame
        if Init == 1:
            Init = 0
            frame_prev = frame

            # Iniciar camera pose rotacion y translacion
            R_prev = R0
            t_prev = t0

            print("Iteracion: ", iter, " - Inicializacion")
            iter = iter + 1

        else:
            print("Iteracion: ", iter)
            # Extraer features
            pts_prev, pts, kp_prev, kp, matches = extract_features(frame_prev, frame, K, max_features)
            print("Número de keypoints detectados en imagen 1 despues del match: ", len(pts_prev))
            print("Número de keypoints detectados en imagen 2 despues del match: ", len(pts))

            # Filtrar matches (RANSAC) y obtener Essential Matrix
            pts_prev_inlier, pts_inlier, mask, E = filter_points_ransac(pts_prev, pts, K,'essential', 1.0)
            print("Número de keypoints detectados en imagen 1 despues de RANSAC: ", len(pts_prev_inlier))
            print("Número de keypoints detectados en imagen 2 despues de RANSAC: ", len(pts_inlier))

            # Obtener inliers matches y crear imagen con los matches
            matchesMask = mask.ravel().tolist()
            inlier_matches = [m for i, m in enumerate(matches) if matchesMask[i] == 1]
            img_matches_inliers = cv2.drawMatches(frame_prev, kp_prev, frame, kp, inlier_matches, None,
                                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            # Recuperar camera pose relativa entre los dos frames
            # cambiado orden de las imagenes en los inputs, lo mismo en el calculo de los 3d
            retval, R_rel, t_rel, mask = recover_camera_pose(E, pts_inlier, pts_prev_inlier, K)

            retval_test, R_test, t_test, _ = cv2.recoverPose(E, pts_inlier, pts_prev_inlier, K)

            # Actualizar camera pose abosluta
            # TODO
            T_relative = np.eye(4)
            T_relative[:3, :3] = R_rel
            T_relative[:3, 3] = t_rel.flatten()
            previous_pose = np.eye(4)
            previous_pose[:3, :3] = R_prev
            previous_pose[:3, 3] = t_prev.flatten()
            T_absolute = previous_pose @ T_relative  # Matrix multiplication
            R_world_to_camera = T_absolute[:3, :3]
            t_world_to_camera = T_absolute[:3, 3]

            # Extraer los puntos en 3D
            # points_3d = Extraer3D_points(R_prev, t_prev, R_rel, t_rel, pts_prev_inlier, pts_inlier, K)
            points_3d = Extraer3D_points(R_prev, t_prev, R_rel, t_rel, pts_inlier, pts_prev_inlier, K)

            # Check 3d points in the plot
            # plot_3d_points(points_3d)
            # plot_3d_points_and_cameras(points_3d, R_rel, t_rel, K)
            # plt.show()

            # Update the plot incrementally
            new_position = np.array(t_world_to_camera)
            new_points = points_3d
            update_plot(new_points, new_position)

            # Mostrar y guardar imagenes
            plt.figure("test2")
            plt.imshow(img_matches_inliers)
            plt.title(f'Iteracion {iter} - Inlier Matches (RANSAC) - Keypoints: {len(pts_inlier)}')
            # plt.title(" Inlier Matches (RANSAC) - Keypoints: " + str(len(pts_inlier)))
            plt.axis('off')
            file_name = f'Matches Iteracion - {iter}.png'
            plt.savefig(file_name)

            # Actualizar para la siguiente iteracion
            frame_prev = frame
            iter = iter + 1

plt.show()

# Now `images` contains all the loaded images
print(f"Loaded {len(images)} images from the path:  {folder_path}" )




