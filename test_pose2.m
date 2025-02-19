% Load camera intrinsic parameters
% load('cameraParams.mat'); % This should contain cameraParams (intrinsic parameters)
load('/home/efren/PycharmProjects/TFM/Calibrar/CameraCalibration_3coeff_movil.mat');
% Simulate two camera poses
% Camera 1 is at the origin
R1 = eye(3); % Rotation matrix for the first camera (identity, no rotation)
t1 = [0; 0; 0]; % Translation vector for the first camera (at origin)

% Camera 2 is translated and rotated relative to Camera 1
R2 = R1; % Rotation matrix for the second camera (10 degrees around X, 5 degrees around Y)
t2 = [0.5; 0.1; 0.2]; % Translation vector for the second camera (not at origin)

% Create camera matrices
P1 = cameraMatrix(cameraParams, R1, t1); % Camera matrix for the first view
P2 = cameraMatrix(cameraParams, R2, t2); % Camera matrix for the second view

% Generate a set of 3D points in the world
numPoints = 50;
worldPoints = rand(numPoints, 3) * 10; % Random 3D points in a 10x10x10 cube

% Project the 3D points into the two camera views
imagePoints1 = worldToImage(cameraParams, R1, t1, worldPoints);
imagePoints2 = worldToImage(cameraParams, R2, t2, worldPoints);

% Add some noise to the image points to simulate real-world conditions
noiseLevel = 1; % Noise level in pixels
imagePoints1 = imagePoints1 + randn(size(imagePoints1)) * noiseLevel;
imagePoints2 = imagePoints2 + randn(size(imagePoints2)) * noiseLevel;

% Triangulate the 3D points from the two views
reconstructedPoints = triangulate(imagePoints1, imagePoints2, P1, P2);

% Display the results
figure;
subplot(1, 2, 1);
plot3(worldPoints(:, 1), worldPoints(:, 2), worldPoints(:, 3), 'b.');
hold on;
plotCamera('Location', t1, 'Orientation', R1, 'Size', 0.2, 'Color', 'r');
plotCamera('Location', t2, 'Orientation', R2, 'Size', 0.2, 'Color', 'g');
axis equal;
xlabel('X');
ylabel('Y');
zlabel('Z');
title('Ground Truth 3D Points and Camera Poses');
grid on;
legend('3D Points', 'Camera 1', 'Camera 2');

subplot(1, 2, 2);
plot3(reconstructedPoints(:, 1), reconstructedPoints(:, 2), reconstructedPoints(:, 3), 'b.');
hold on;
plotCamera('Location', t1, 'Orientation', R1, 'Size', 0.2, 'Color', 'r');
plotCamera('Location', t2, 'Orientation', R2, 'Size', 0.2, 'Color', 'g');
axis equal;
xlabel('X');
ylabel('Y');
zlabel('Z');
title('Reconstructed 3D Points and Camera Poses');
grid on;
legend('Reconstructed Points', 'Camera 1', 'Camera 2');