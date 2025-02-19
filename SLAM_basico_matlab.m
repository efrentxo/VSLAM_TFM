%% Load data
clc; clear; close all;
% imageFolder   = ['/home/efren/Escritorio/TFM/datos/Crazyflie_test/'];
% imageFolder   = ['/home/efren/Escritorio/TFM/datos/test1/'];
imageFolder   = ['//home/efren/Escritorio/TFM/datos/Camara_movil_test2//'];

imds          = imageDatastore(imageFolder);
disp(['Imagenes cargadas: ', num2str(numel(imds.Files))])

%% Display settings
plottest = 0; 

%%
% Inspect the first image
disp ('Analizando primera imagen...')
currFrameIdx = 1;
currI = readimage(imds, currFrameIdx);
% figure;
% himage = imshow(currI);
% title('Imagen - 1')

% load('/home/efren/Escritorio/TFM/Calibrar Crazyflie/CrazyFlie_CameraCalibration06022025_3coeff.mat')
load('/home/efren/PycharmProjects/TFM/Calibrar/CameraCalibration_3coeff_movil.mat');
% cameraParams_3coeff = cameraParams_movil;
% cameraParams = cameraParams_3coeff;

focalLength    = cameraParams.FocalLength;    % in units of pixels
principalPoint = cameraParams.PrincipalPoint;    % in units of pixels
imageSize      = size(currI,[1 2]);  % in units of pixels
RadialDistorsion  = [cameraParams.RadialDistortion];
RadialDistorsion  = [0 0 0];
TangencialDistorsion = cameraParams.TangentialDistortion;
intrinsics     = cameraIntrinsics(focalLength, principalPoint, imageSize, "RadialDistortion",RadialDistorsion, "TangentialDistortion",TangencialDistorsion);
disp(['Images resolucion: ', num2str(imageSize(1)),'x',num2str(imageSize(2))]);
disp('Camera settings loaded');

% Check camera checkings and image resolution match
if imageSize(2)~= cameraParams.ImageSize(1) ||  imageSize(1)~= cameraParams.ImageSize(2)
    disp('Los settings de calibracion no cuadran con las imagenes.');
end

% Set random seed for reproducibility
rng(0);

% Detect and extract ORB features
scaleFactor = 1.2;
numLevels   = 8;
numPoints   = 500;
% Corregir imagen
currI_corr  = undistortImage(currI, intrinsics);
[preFeatures, prePoints, preAllPoints] = DetectAndExtractFeatures(currI_corr, scaleFactor, numLevels, numPoints, intrinsics); 

% Initialize Pose
R_ini = eye(3);
t_ini = zeros(3,1);
trajectory = t_ini';
structure3D = [];
Pose_track(1) = rigidtform3d(R_ini, t_ini);

currFrameIdx = currFrameIdx + 1;
firstI_corr = currI_corr; % Preserve the first frame 
preI_corr = firstI_corr; 

figure; 
subplot(121)
himage = imshow(currI);
title (['Imagen Original'])
subplot(122)
himage = imshow(currI_corr);
hold on
plot(prePoints, 'ShowScale',false)
title (['Imagen corregida + ORB points (',num2str(size(prePoints,1)),')'])

disp ('Analizando primera imagen...DONE')

%% Process Frames
% Map initialization loop
disp ('Analizando siguientes frames...')
t_pre= t_ini;
R_pre= R_ini;
Pose_pre = rigidtform3d(R_ini,t_ini);
key_frame = 2;

% no tengo claro la diferencia con la funcion pose2extr 
%
while currFrameIdx <= numel(imds.Files)
    currI = readimage(imds, currFrameIdx);
    disp (['Imagen - ',num2str(currFrameIdx),'...'])

    % Corregir imagen
    currI_corr  = undistortImage(currI, intrinsics);
    figure;
    himage = imshow(currI);
    title(['Imagen - ',num2str(currFrameIdx)])

    % Detect and Match Features and apply camera distorsion and K
    [currFeatures, currPoints] = DetectAndExtractFeatures(currI_corr, scaleFactor, numLevels, numPoints); 
    disp ([' -- Puntos ORB extraidos... ',num2str(size(currPoints,1))])

%     indexPairs = matchFeatures(preFeatures, currFeatures);
    clear indexPairs
    indexPairs = matchFeatures(preFeatures, currFeatures, Unique=true, MaxRatio=0.7, MatchThreshold=20);
    clear matchedPointsPrev matchedPointsCurr
    matchedPointsPrev = prePoints(indexPairs(:,1));
    matchedPointsCurr = currPoints(indexPairs(:,2));
    
    disp ([' -- Match Puntos ORB... ',num2str(size(matchedPointsPrev,1))])
    if size(matchedPointsPrev,1) < 10
%         continue;
    end
    
    % Obtener Essential matrix appliying RANSAC
    clear inliers
    [E, inliers] = estimateEssentialMatrix(matchedPointsPrev, matchedPointsCurr, cameraParams,'MaxNumTrials', 500, 'Confidence', 99.9, 'MaxDistance', 1);
    
    InliersPoints = sum(inliers);
    disp ([' -- Inliers points: ',num2str(InliersPoints)])

    %% Check minimum inliers?
    %% TODO

    % Display inliers
    figure;
    showMatchedFeatures(preI_corr, currI_corr, matchedPointsPrev(inliers,:), matchedPointsCurr(inliers,:), 'montage');
    title(['Inlier Matches after RANSAC - ',num2str(size(matchedPointsPrev(inliers,:),1)), ' points']);
    figure;
    showMatchedFeatures(preI_corr, currI_corr, matchedPointsPrev(inliers,:), matchedPointsCurr(inliers,:), 'blend');
    title(['Inlier Matches after RANSAC - ',num2str(size(matchedPointsPrev(inliers,:),1)), ' points']);

    % Estimate Motion using PnP
    % [E, inliers] = estimateEssentialMatrix(matchedPointsPrev.Location, matchedPointsCurr.Location, cameraParams);
%     [R_new, t_new] = relativeCameraPose(E, cameraParams, matchedPointsPrev(inliers), matchedPointsCurr(inliers));
    [relPose, validFraction] = estrelpose(E, cameraParams.Intrinsics, matchedPointsPrev(inliers), matchedPointsCurr(inliers));

     % TODO investigar cuando se puede usar "estworldpose"
%     [currPose, inlier, status] = estworldpose(matchedImagePoints, matchedWorldPoints, intrinsics, ...
%     'Confidence', 95, 'MaxReprojectionError', 3, 'MaxNumTrials', 1e4);

    % Triangulate 3D Points
%     M1 = cameraParams.IntrinsicMatrix' * [R, t];
%     M2 = cameraParams.IntrinsicMatrix' * [R * R_new', t + R * t_new'];
%     points3D = triangulate(matchedPointsPrev(inliers), matchedPointsCurr(inliers), M1', M2');
%     structure3D = [structure3D; points3D];

    % Update Pose
    t_curr = t_pre + R_pre * relPose.Translation';
%     R_curr = R_pre * relPose.R';% incorect
    R_curr = R_pre * relPose.R; % correct
    Pose_curr = rigidtform3d(R_curr, t_curr);
    trajectory = [trajectory; t_curr'];
    pose_track(key_frame) = Pose_curr;

    % Triangulate two views to obtain 3-D map points
    %%%% TODO fix issue with previous pose
    minParallax = 1; % In degrees
    [isValid, xyzWorldPoints, inlierTriangulationIdx] = helperTriangulateTwoFrames(...
        Pose_pre, Pose_curr, matchedPointsPrev(inliers), matchedPointsCurr(inliers), cameraParams.Intrinsics, minParallax);
    inlinersMatchedPointsCurr = matchedPointsCurr(inliers);
    inlinersmatchedPointsPrev = matchedPointsPrev(inliers);

    % add pounts to the main 
    structure3D = [structure3D; xyzWorldPoints];        
    
    % Get the original index of features in the two key frames
    trianInlinersMatchedPointsCurr = inlinersMatchedPointsCurr(inlierTriangulationIdx);
    trianInlinersmatchedPointsPrev = inlinersmatchedPointsPrev(inlierTriangulationIdx);

    % Update Keyframe
    key_frame = key_frame + 1;

    % Plot Trajectory and 3D Points
    h = figure;
%     subplot(2,2,1);
    plotCameraPose(R_pre,t_pre, 1.5);hold on
    plotCameraPose(R_curr,t_curr, 1.5) 
%     cam = plotCamera(AbsolutePose=Pose_pre,Opacity=0) %% otra opcion para
%     plotear la camara.. revisar
%     plotCameraPose(relPose.R,relPose.Translation', 1.5) % lo mismo 
    title('SLAM 3d points + camera pose');
    hold on;
    scatter3(structure3D(:,1), structure3D(:,2), structure3D(:,3), 1, 'b');
    scatter3(structure3D(1,1), structure3D(1,2), structure3D(1,3), 1, 'r+','LineWidth',10);
    hold on;
    grid on;
    xlabel('X'); ylabel('Y'); zlabel('Z');
    title('Monocular SLAM Trajectory');
    
    %figure;
    %% Test
    if plottest
        for i=1:length(trianInlinersMatchedPointsCurr)
            figure;
            subplot(2,2,1);
            plotCameraPoseXY(R_pre,t_pre, 1);hold on
            plotCameraPoseXY(R_curr,t_curr, 1) 
            title('SLAM 3d points + camera pose');
            hold on;
            plot(structure3D(:,1), structure3D(:,2), 'b.');
            plot(structure3D(i,1), structure3D(i,2), 'r+','LineWidth',2);
            set(gca, 'YDir','reverse')
            hold on;
            grid on;
            xlabel('X'); ylabel('Y'); zlabel('Z');
            title('Front view');
    
            subplot(2,2,3)
            plot(structure3D(:,1), structure3D(:,3), 'b.'); hold on;
            plot(structure3D(i,1), structure3D(i,3), 'r+','LineWidth',2);
            plotCameraPoseXZ(R_pre,t_pre, 1);hold on
            plotCameraPoseXZ(R_curr,t_curr, 1) 
            hold on;
            grid on;
            xlabel('X'); ylabel('Z'); zlabel('Z');
            title('Top view')
    
            subplot(2,2,2)
            himage = imshow(currI_corr);
            hold on
            plot(trianInlinersMatchedPointsCurr, 'ShowScale',false)
            hold on
            title (['Imagen corregida + punto 3d (',num2str(i),'/',num2str(size(trianInlinersMatchedPointsCurr,1)),')'])
    
            subplot(2,2,4)
            himage = imshow(currI_corr);
            hold on
            plot(trianInlinersMatchedPointsCurr(i), 'ShowScale',false)
            hold on
            title (['Imagen corregida + punto 3d (',num2str(i),'/',num2str(size(trianInlinersMatchedPointsCurr,1)),')'])
        
        pause
        end
    end

    % Update Previous Frame Features
    preFrame    = currI;
    prePoints   = currPoints;
    preFeatures = currFeatures;
    preI_corr   = currI_corr;
    currFrameIdx = currFrameIdx + 1;
    t_pre = t_curr;
    R_pre = R_curr;
    Pose_pre = Pose_curr;

    %%
    pause(0.01);
end


%% PLOTS FINAL
plot_info = 1; 
if plot_info
    figure;

    %% TODO 
    % plot numero puntos ORB detectadas
    % plot numero matches
    % plot numero inliers
    % vs numero de frame/iteracion
end
    