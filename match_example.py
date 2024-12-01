import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load two consecutive frames (example: synthetic frames with motion)
img1_path = 'imagen1.jpg'
frame1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
img2_path = 'imagen2.jpg'
frame2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)


# Add features to the frames
# cv2.circle(frame1, (150, 200), 10, 255, -1)
# cv2.circle(frame2, (160, 210), 10, 255, -1)  # Simulate motion
# cv2.rectangle(frame1, (250, 150), (270, 170), 255, -1)
# cv2.rectangle(frame2, (260, 160), (280, 180), 255, -1)

# Detect features in the first frame (Shi-Tomasi corner detection)
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
p0 = cv2.goodFeaturesToTrack(frame1, mask=None, **feature_params)

# Calculate optical flow using Lucas-Kanade method
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
p1, st, err = cv2.calcOpticalFlowPyrLK(frame1, frame2, p0, None, **lk_params)

# Filter only the points with valid status
good_old = p0[st == 1]
good_new = p1[st == 1]

# Visualize the motion vectors
motion_image = cv2.cvtColor(frame1, cv2.COLOR_GRAY2BGR)
for i, (new, old) in enumerate(zip(good_new, good_old)):
    x_new, y_new = new.ravel()
    x_old, y_old = old.ravel()
    motion_image = cv2.line(motion_image, (int(x_new), int(y_new)), (int(x_old), int(y_old)), (0, 255, 0), 2)
    motion_image = cv2.circle(motion_image, (int(x_new), int(y_new)), 5, (0, 0, 255), -1)

# Display the motion estimation result
plt.figure(figsize=(8, 8))
plt.imshow(motion_image)
plt.title("Motion Estimation with Optical Flow")
plt.axis('off')
plt.show()
