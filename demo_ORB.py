import cv2
import numpy as np
#import matplotlib as plt
#from matplotlib import pyplot as plt
import matplotlib.pyplot as plt

opencv2_plots_raw = 0
opencv2_plots_keypoints = 0
opencv2_plots_match = 0

# Load the image
img1_path = 'imagen1.jpg'
img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)

img2_path = 'imagen2.jpg'
img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

if opencv2_plots_raw:
    # Mostrar imagenes
    cv2.startWindowThread()
    cv2.imshow('Image 1 - raw', img1)
    cv2.imshow('Image 2 - raw', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Initialize ORB detector
ORB_features = 500;
orb = cv2.ORB_create(nfeatures=ORB_features)

# Detect keypoints and descriptors
keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

# Imprimir el número de keypoints detectados
print("Número de keypoints detectados en imagen 1:", len(keypoints1))
# Imprimir el número de keypoints detectados
print("Número de keypoints detectados en imagen 2:", len(keypoints2))

# Inspeccionar el tamaño de los descriptores (si se calculan)
if descriptors1 is not None:
    print("Tamaño del array de descriptores en imagen 1:", descriptors1.shape)
else:
    print("No se detectaron descriptores en imagen 1.")

# Inspeccionar el tamaño de los descriptores (si se calculan)
if descriptors2 is not None:
    print("Tamaño del array de descriptores en imagen 1:", descriptors2.shape)
else:
    print("No se detectaron descriptores en imagen 1.")

# Draw keypoints on the images
img1_kp = cv2.drawKeypoints(img1, keypoints1, None, color=(0, 255, 0), flags=0)
img2_kp = cv2.drawKeypoints(img2, keypoints2, None, color=(0, 255, 0), flags=0)

if opencv2_plots_keypoints:
    # Show the images with keypoints
    cv2.imshow('Keypoints - Image 1', img1_kp)
    cv2.imshow('Keypoints - Image 2', img2_kp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Match descriptors from current frame with previous frame
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)

# Sort matches based on distance (lower distance is better)
matches = sorted(matches, key=lambda x: x.distance)

# Draw matches
img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

if opencv2_plots_match:
    # Show the matched keypoints
    cv2.imshow("Matches", img_matches)
    cv2.waitKey(0)

# Mostrar imagenes
if 1 == 1:
    # Display the keypoint image
    plt.figure(figsize=(10, 6))
    plt.imshow(img_matches, cmap='gray')
    plt.title("Matches - " + str(len(matches[:10])))
    plt.axis('off')
    plt.show()

    # Display the keypoint image 1
    plt.figure(figsize=(10, 6))
    plt.imshow(img1_kp, cmap='gray')
    plt.title("ORB Keypoints - Image 1 - Keypoints: " + str(len(keypoints1)))
    plt.axis('off')
    plt.show()

    # Display the keypoint image 2
    plt.figure(figsize=(10, 6))
    plt.imshow(img2_kp, cmap='gray')
    plt.title("ORB  Keypoints - Image 2 - Keypoints: " + str(len(keypoints2)))
    plt.axis('off')
    plt.show()

# TO DO

# Extract matched points
pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

# Estimate homography using RANSAC
H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
matchesMask = mask.ravel().tolist()

# Visualize matches before RANSAC
img_matches_all = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Visualize matches after RANSAC
inlier_matches = [m for i, m in enumerate(matches) if matchesMask[i] == 1]
img_matches_inliers = cv2.drawMatches(img1, keypoints1, img2, keypoints2, inlier_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Plot results
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.imshow(img_matches_all)
plt.title("All Matches - Keypoints: " + str(len(img_matches_all)))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_matches_inliers)
plt.title("Inlier Matches (RANSAC) - Keypoints: " + str(len(inlier_matches)))
plt.axis('off')

plt.show()
