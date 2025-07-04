% Auto-generated by cameraCalibrator app on 08-Feb-2025
%-------------------------------------------------------


% Define images to process
imageFileNames = {'/home/efren/PycharmProjects/TFM/Calibrar Crazyflie/img_000032.png',...
    '/home/efren/PycharmProjects/TFM/Calibrar Crazyflie/img_000034.png',...
    '/home/efren/PycharmProjects/TFM/Calibrar Crazyflie/img_000035.png',...
    '/home/efren/PycharmProjects/TFM/Calibrar Crazyflie/img_000042.png',...
    '/home/efren/PycharmProjects/TFM/Calibrar Crazyflie/img_000044.png',...
    '/home/efren/PycharmProjects/TFM/Calibrar Crazyflie/img_000050.png',...
    '/home/efren/PycharmProjects/TFM/Calibrar Crazyflie/img_000051.png',...
    '/home/efren/PycharmProjects/TFM/Calibrar Crazyflie/img_000064.png',...
    '/home/efren/PycharmProjects/TFM/Calibrar Crazyflie/img_000065.png',...
    '/home/efren/PycharmProjects/TFM/Calibrar Crazyflie/img_000066.png',...
    '/home/efren/PycharmProjects/TFM/Calibrar Crazyflie/img_000067.png',...
    '/home/efren/PycharmProjects/TFM/Calibrar Crazyflie/img_000068.png',...
    '/home/efren/PycharmProjects/TFM/Calibrar Crazyflie/img_000071.png',...
    '/home/efren/PycharmProjects/TFM/Calibrar Crazyflie/img_000072.png',...
    '/home/efren/PycharmProjects/TFM/Calibrar Crazyflie/img_000083.png',...
    '/home/efren/PycharmProjects/TFM/Calibrar Crazyflie/img_000089.png',...
    '/home/efren/PycharmProjects/TFM/Calibrar Crazyflie/img_000106.png',...
    '/home/efren/PycharmProjects/TFM/Calibrar Crazyflie/img_000108.png',...
    '/home/efren/PycharmProjects/TFM/Calibrar Crazyflie/img_000110.png',...
    '/home/efren/PycharmProjects/TFM/Calibrar Crazyflie/img_000116.png',...
    '/home/efren/PycharmProjects/TFM/Calibrar Crazyflie/img_000140.png',...
    '/home/efren/PycharmProjects/TFM/Calibrar Crazyflie/img_000144.png',...
    '/home/efren/PycharmProjects/TFM/Calibrar Crazyflie/img_000147.png',...
    '/home/efren/PycharmProjects/TFM/Calibrar Crazyflie/img_000148.png',...
    '/home/efren/PycharmProjects/TFM/Calibrar Crazyflie/img_000155.png',...
    '/home/efren/PycharmProjects/TFM/Calibrar Crazyflie/img_000156.png',...
    '/home/efren/PycharmProjects/TFM/Calibrar Crazyflie/img_000166.png',...
    '/home/efren/PycharmProjects/TFM/Calibrar Crazyflie/img_000167.png',...
    '/home/efren/PycharmProjects/TFM/Calibrar Crazyflie/img_000168.png',...
    };
% Detect calibration pattern in images
detector = vision.calibration.monocular.CheckerboardDetector();
[imagePoints, imagesUsed] = detectPatternPoints(detector, imageFileNames);
imageFileNames = imageFileNames(imagesUsed);

% Read the first image to obtain image size
originalImage = imread(imageFileNames{1});
[mrows, ncols, ~] = size(originalImage);

% Generate world coordinates for the planar pattern keypoints
squareSize = 23;  % in units of 'millimeters'
worldPoints = generateWorldPoints(detector, 'SquareSize', squareSize);

% Calibrate the camera
[cameraParams, imagesUsed, estimationErrors] = estimateCameraParameters(imagePoints, worldPoints, ...
    'EstimateSkew', false, 'EstimateTangentialDistortion', true, ...
    'NumRadialDistortionCoefficients', 2, 'WorldUnits', 'millimeters', ...
    'InitialIntrinsicMatrix', [], 'InitialRadialDistortion', [], ...
    'ImageSize', [mrows, ncols]);

% View reprojection errors
h1=figure; showReprojectionErrors(cameraParams);

% Visualize pattern locations
h2=figure; showExtrinsics(cameraParams, 'CameraCentric');

% Display parameter estimation errors
displayErrors(estimationErrors, cameraParams);

% For example, you can use the calibration data to remove effects of lens distortion.
undistortedImage = undistortImage(originalImage, cameraParams);

% See additional examples of how to use the calibration data.  At the prompt type:
% showdemo('MeasuringPlanarObjectsExample')
% showdemo('StructureFromMotionExample')
