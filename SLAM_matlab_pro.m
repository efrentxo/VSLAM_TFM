%% Load data
clc; clear; close all;
% imageFolder   = ['/home/efren/Escritorio/TFM/datos/Crazyflie_test/'];
imageFolder   = ['//home/efren/Escritorio/TFM/datos/Crazyflie_toma1//'];
% imageFolder   = ['//home/efren/Escritorio/TFM/datos/Camara_movil_test2//'];
% imageFolder   = ['//home/efren/Escritorio/TFM/datos/Camara_movil_test5//'];
% 
imds          = imageDatastore(imageFolder);
disp(['Imagenes cargadas: ', num2str(numel(imds.Files))])

% crear subfolder para guardar outputs
NewSubFolder= 'Output';
if ~exist(fullfile(imageFolder,NewSubFolder))
    mkdir(fullfile(imageFolder,NewSubFolder));
end
pathOutputs = (fullfile(imageFolder,NewSubFolder));

% seleccionar si generar plots para guardar
CrearPlots = 1;

%% Iniciar mapa 3D
% leer la primera imagen
disp ('Analizando primera imagen...')
currFrameIdx = 1;
currI = readimage(imds, currFrameIdx);

% Cargar coeficientes de calibracion de la camara
load('/home/efren/Escritorio/TFM/Calibrar Crazyflie/CrazyFlie_CameraCalibration06022025_3coeff.mat')
% load('/home/efren/PycharmProjects/TFM/Calibrar/CameraCalibration_3coeff_movil.mat');
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

% Corregir imagen
currI_corr{currFrameIdx}  = undistortImage(currI, intrinsics);

% Detect and extract ORB features
scaleFactor = 1.2;
numLevels   = 8;
numPoints   = 500;
[preFeatures, prePoints, preAllPoints] = DetectAndExtractFeatures(currI_corr{currFrameIdx}, scaleFactor, numLevels, numPoints, intrinsics); 

currFrameIdx = currFrameIdx + 1;
firstI       = currI; % Preserve the first frame 

isMapInitialized  = false;

disp ('Analizando primera imagen...DONE')

%% Map initialization loop
while ~isMapInitialized && currFrameIdx < numel(imds.Files)
    disp (['presiona cualquier tecla para continuar...'])
    pause;
    currI = readimage(imds, currFrameIdx);
    disp (['Imagen - ',num2str(currFrameIdx),'...'])

    % Corregir imagen
    currI_corr{currFrameIdx}  = undistortImage(currI, intrinsics);

    [currFeatures, currPoints] = DetectAndExtractFeatures(currI_corr{currFrameIdx} , scaleFactor, numLevels, numPoints); 

    currFrameIdx = currFrameIdx + 1;

    % Find putative feature matches
    indexPairs = matchFeatures(preFeatures, currFeatures, Unique=true, MaxRatio=0.9, MatchThreshold=40);

    % If not enough matches are found, check the next frame
%     minMatches = 100;
%     if size(indexPairs, 1) < minMatches
%         continue
%     end

    preMatchedPoints  = prePoints(indexPairs(:,1),:);
    currMatchedPoints = currPoints(indexPairs(:,2),:);

%     % Display matches
%     figure;
%     showMatchedFeatures(firstI , currI_corr{currFrameIdx-1} , preMatchedPoints, currMatchedPoints, 'montage');
%     title(['Matches - ',num2str(currMatchedPoints.Count), ' points']);

%     % Compute homography and evaluate reconstruction
%     [tformH, scoreH, inliersIdxH] = helperComputeHomography(preMatchedPoints, currMatchedPoints);
% 
%     % Compute fundamental matrix and evaluate reconstruction
%     [tformF, scoreF, inliersIdxF] = helperComputeFundamentalMatrix(preMatchedPoints, currMatchedPoints, intrinsics);
% 
%     % Select the model based on a heuristic
%     ratio = scoreH/(scoreH + scoreF);
%     ratioThreshold = 0.45;
%     if ratio > ratioThreshold
%         inlierTformIdx = inliersIdxH;
%         tform          = tformH;
%     else
%         inlierTformIdx = inliersIdxF;
%         tform          = tformF;
%     end

    % Obtener Essential matrix appliying RANSAC
    clear inliersIdx
    [E, inliersIdx] = estimateEssentialMatrix(preMatchedPoints, currMatchedPoints, cameraParams,'MaxNumTrials', 500, 'Confidence', 99.9, 'MaxDistance', 1);
    
    inlierTformIdx = inliersIdx;
    tform = E;
    % Computes the camera location up to scale. Use half of the 
    % points to reduce computation
    inlierPrePoints  = preMatchedPoints(inlierTformIdx);
    inlierCurrPoints = currMatchedPoints(inlierTformIdx);
    indexPairs = indexPairs(inliersIdx,:);

%     % Display Matches & Inliers 
%     figure;
%     subplot(211)
%     showMatchedFeatures(firstI, currI_corr{currFrameIdx-1}, preMatchedPoints, currMatchedPoints, 'montage');
%     title(['Matches - ',num2str(currMatchedPoints.Count), ' points']);
% 
%     subplot(212);
%     showMatchedFeatures(firstI, currI_corr{currFrameIdx-1}, inlierPrePoints, inlierCurrPoints, 'montage');
%     title(['Inliers - ',num2str(inlierPrePoints.Count), ' points']);

    [relPose, validFraction] = estrelpose(tform, intrinsics, inlierPrePoints(1:2:end), inlierCurrPoints(1:2:end));

    % If not enough inliers are found, move to the next frame
    if validFraction < 0.9 || numel(relPose)>1
        disp (['Imagen - ',num2str(currFrameIdx-1),'...no sufientes inliers']);
        continue
    end

    % Triangulate two views to obtain 3-D map points
    minParallax = 5; % In degrees
    [isValid, xyzWorldPoints, inlierTriangulationIdx] = helperTriangulateTwoFrames(...
        rigidtform3d, relPose, inlierPrePoints, inlierCurrPoints, intrinsics, minParallax);
    
    % Q&D
    isValid = 1;
    if ~isValid
        disp (['Imagen - ',num2str(currFrameIdx-1),'...triangulacion no valida']);
        continue
    end

    inlierPrePoints  = inlierPrePoints(inlierTriangulationIdx);
    inlierCurrPoints = inlierCurrPoints(inlierTriangulationIdx);

    % Display Matches & Final Inliers 
    f1 = figure;
    subplot(211)
    showMatchedFeatures(firstI, currI_corr{currFrameIdx-1}, preMatchedPoints, currMatchedPoints, 'montage');
    title(['Matches - ',num2str(currMatchedPoints.Count), ' points']);

    subplot(212);
    showMatchedFeatures(firstI, currI_corr{currFrameIdx-1}, inlierPrePoints, inlierCurrPoints, 'montage');
    title(['Inliers - ',num2str(inlierPrePoints.Count), ' points']);

    % Get the original index of features in the two key frames
    indexPairs = indexPairs(inlierTriangulationIdx,:);
    
    % Display Final Inliers check
    figure;
    showMatchedFeatures(firstI , currI_corr{currFrameIdx-1} , prePoints(indexPairs(:,1)), currPoints(indexPairs(:,2)), 'blend');
    title(['Inliers - ',num2str(inlierPrePoints.Count), ' points']);

    isMapInitialized = true;

    disp(['Map initialized with frame 1 and frame ', num2str(currFrameIdx-1)])
end % End of map initialization loop

%% Iniciar mapa3d y views 
% Create an empty imageviewset object to store key frames
disp(['Almacenar mapa 3d y camera pose... '])
vSetKeyFrames = imageviewset;

% Create an empty worldpointset object to store 3-D map points
mapPointSet   = worldpointset;

% Add the first key frame. Place the camera associated with the first 
% key frame at the origin, oriented along the Z-axis
preViewId     = 1;
vSetKeyFrames = addView(vSetKeyFrames, preViewId, rigidtform3d, Points=prePoints,...
    Features=preFeatures.Features);

% Add the second key frame
currViewId    = 2;
vSetKeyFrames = addView(vSetKeyFrames, currViewId, relPose, Points=currPoints,...
    Features=currFeatures.Features);

% Add connection between the first and the second key frame
vSetKeyFrames = addConnection(vSetKeyFrames, preViewId, currViewId, relPose, Matches=indexPairs);

% Add 3-D map points
[mapPointSet, newPointIdx] = addWorldPoints(mapPointSet, xyzWorldPoints);

% Add image points corresponding to the map points in the first key frame
mapPointSet   = addCorrespondences(mapPointSet, preViewId, newPointIdx, indexPairs(:,1));

% Add image points corresponding to the map points in the second key frame
mapPointSet   = addCorrespondences(mapPointSet, currViewId, newPointIdx, indexPairs(:,2));

disp(['Almacenar mapa 3d y camera pose... DONE'])

%%TODO bundle adjustment... 
disp(['Aplicar bundle adjustment... '])
% Run full bundle adjustment on the first two key frames
tracks       = findTracks(vSetKeyFrames);
cameraPoses  = poses(vSetKeyFrames);

[refinedPoints, refinedAbsPoses] = bundleAdjustment(xyzWorldPoints, tracks, ...
    cameraPoses, intrinsics, FixedViewIDs=1, ...
    PointsUndistorted=true, AbsoluteTolerance=1e-7,...
    RelativeTolerance=1e-15, MaxIteration=20, ...
    Solver="preconditioned-conjugate-gradient");

% Scale the map and the camera pose using the median depth of map points
medianDepth   = median(vecnorm(refinedPoints.'));
refinedPoints = refinedPoints / medianDepth;

refinedAbsPoses.AbsolutePose(currViewId).Translation = ...
    refinedAbsPoses.AbsolutePose(currViewId).Translation / medianDepth;
relPose.Translation = relPose.Translation/medianDepth;

% Update key frames with the refined poses
vSetKeyFrames = updateView(vSetKeyFrames, refinedAbsPoses);
vSetKeyFrames = updateConnection(vSetKeyFrames, preViewId, currViewId, relPose);

% Update map points with the refined positions
mapPointSet = updateWorldPoints(mapPointSet, newPointIdx, refinedPoints);

% Update view direction and depth 
mapPointSet = updateLimitsAndDirection(mapPointSet, newPointIdx, vSetKeyFrames.Views);

% Update representative view
mapPointSet = updateRepresentativeView(mapPointSet, newPointIdx, vSetKeyFrames.Views);

% Visualize matched features in the current frame
% close(hfeature.Parent.Parent);
featurePlot   = helperVisualizeMatchedFeatures(currI, currPoints(indexPairs(:,2)));

% Visualize initial map points and camera trajectory
mapPlot       = helperVisualizeMotionAndStructure(vSetKeyFrames, mapPointSet);

% Show legend
showLegend(mapPlot);
disp(['Aplicar bundle adjustment...DONE '])
%% plot 3d inicial

%% Tracking
% ViewId of the current key frame
disp (['Leyendo el resto de imagenes... '])
currKeyFrameId   = currViewId;

% ViewId of the last key frame
lastKeyFrameId   = currViewId;

% Index of the last key frame in the input image sequence
lastKeyFrameIdx  = currFrameIdx - 1; 

% Indices of all the key frames in the input image sequence
addedFramesIdx   = [1; lastKeyFrameIdx];

isLoopClosed     = false;


% Loop principal
isLastFrameKeyFrame = true;

% Create and initialize the KLT tracker
% tracker = vision.PointTracker(MaxBidirectionalError = 5);
% initialize(tracker, currPoints.Location(indexPairs(:,2), :), currI);

while currFrameIdx < numel(imds.Files)  
    disp(['Imagen iteracion ', num2str(currFrameIdx)])    
    currI = readimage(imds, currFrameIdx);
    % Corregir imagen
    currI_corr{currFrameIdx}  = undistortImage(currI, intrinsics);
    
%     figure;
%     imshow(currI_corr{currFrameIdx-1})
    [currFeatures, currPoints] = DetectAndExtractFeatures(currI_corr{currFrameIdx} , scaleFactor, numLevels, numPoints);

    f1 = figure;
    [filepath,name,ext] = fileparts(imds.Files(currFrameIdx));
%     imshow(currI_corr{currFrameIdx-1})
    imshow(currI_corr{currFrameIdx});hold on;
    plot(currPoints,'ShowScale',false, showOrientation=false)
    title(['All ORB points - ',num2str(currPoints.Count), ' points']);
    saveas(f1,[pathOutputs,'/All_ORB_points',name,'.png']) ;
    close(f1);


    % Track the last key frame
    [currPose, mapPointsIdx, featureIdx] = helperTrackLastKeyFrame(...
    mapPointSet, vSetKeyFrames.Views, currFeatures, currPoints, lastKeyFrameId, intrinsics, scaleFactor);

    % Track the local map and check if the current frame is a key frame.
    % A frame is a key frame if both of the following conditions are satisfied:
    %
    % 1. At least 20 frames have passed since the last key frame or the
    %    current frame tracks fewer than 100 map points.
    % 2. The map points tracked by the current frame are fewer than 90% of
    %    points tracked by the reference key frame.
    %
    % Tracking performance is sensitive to the value of numPointsKeyFrame.  
    % If tracking is lost, try a larger value.
    %
    % localKeyFrameIds:   ViewId of the connected key frames of the current frame

    % TODO
    numSkipFrames     = 20;
    numPointsKeyFrame = 90;
    [localKeyFrameIds, currPose, mapPointsIdx, featureIdx, isKeyFrame] = ...
        helperTrackLocalMap(mapPointSet, vSetKeyFrames, mapPointsIdx, ...
        featureIdx, currPose, currFeatures, currPoints, intrinsics, scaleFactor, numLevels, ...
        isLastFrameKeyFrame, lastKeyFrameIdx, currFrameIdx, numSkipFrames, numPointsKeyFrame);

    % Visualize matched features
%     updatePlot(featurePlot, currI, currPoints(featureIdx));

    isKeyFrame = 1;
    if ~isKeyFrame
        currFrameIdx        = currFrameIdx + 1;
        isLastFrameKeyFrame = false;
        continue
    else
        currKeyFrameId      = currKeyFrameId + 1;
        isLastFrameKeyFrame = true;
    end
    
    % second part
     % Add the new key frame 
    [mapPointSet, vSetKeyFrames] = helperAddNewKeyFrame(mapPointSet, vSetKeyFrames, ...
        currPose, currFeatures, currPoints, mapPointsIdx, featureIdx, localKeyFrameIds);

    % Remove outlier map points that are observed in fewer than 3 key frames
    outlierIdx    = setdiff(newPointIdx, mapPointsIdx);
    if ~isempty(outlierIdx)
        mapPointSet   = removeWorldPoints(mapPointSet, outlierIdx);
    end

    % Create new map points by triangulation
    minNumMatches = 10;
    minParallax   = 3;
    [mapPointSet, vSetKeyFrames, newPointIdx] = helperCreateNewMapPoints(mapPointSet, vSetKeyFrames, ...
        currKeyFrameId, intrinsics, scaleFactor, minNumMatches, minParallax);

    % Local bundle adjustment
    [refinedViews, dist] = connectedViews(vSetKeyFrames, currKeyFrameId, MaxDistance=2);
    refinedKeyFrameIds = refinedViews.ViewId;
    fixedViewIds = refinedKeyFrameIds(dist==2);
    fixedViewIds = fixedViewIds(1:min(10, numel(fixedViewIds)));

    % Refine local key frames and map points
    [mapPointSet, vSetKeyFrames, mapPointIdx] = bundleAdjustment(...
        mapPointSet, vSetKeyFrames, [refinedKeyFrameIds; currKeyFrameId], intrinsics, ...
        FixedViewIDs=fixedViewIds, PointsUndistorted=true, AbsoluteTolerance=1e-7,...
        RelativeTolerance=1e-16, Solver="preconditioned-conjugate-gradient", ...
        MaxIteration=10);

    % Update view direction and depth
    mapPointSet = updateLimitsAndDirection(mapPointSet, mapPointIdx, vSetKeyFrames.Views);

    % Update representative view
    mapPointSet = updateRepresentativeView(mapPointSet, mapPointIdx, vSetKeyFrames.Views);

    % Visualize 3D world points and camera trajectory
    updatePlot(mapPlot, vSetKeyFrames, mapPointSet);

    % Set the feature points to be tracked
    [~, index2d] = findWorldPointsInView(mapPointSet, currKeyFrameId);
%     setPoints(tracker, currPoints.Location(index2d, :));
    
    % Display Matches & Final Inliers 
    % TODO crear figura con los inliers con el ultimo frame y el anterior
%     f2 = figure;
%     subplot(211)
%     inliers
%     showMatchedFeatures(currI_corr{currFrameIdx}, currI_corr{currFrameIdx-1}, preMatchedPoints, currMatchedPoints, 'montage');
%     title(['Matches - ',num2str(currMatchedPoints.Count), ' points']);
%     inlierPrePoints = vSetKeyFrames.Views.Points{3,1}(1)
%     vSetKeyFrames.Connections.Matches{3,1}
%    
%     % Get the matched points from the imageViewSet
%     [matches] = findTracks(vSetKeyFrames, [2, 3]);
%     points1 = matches(:, 1:2);
%     subplot(212);
%     showMatchedFeatures(currI_corr{currFrameIdx}, currI_corr{currFrameIdx-1}, inlierPrePoints, inlierCurrPoints, 'montage');
%     title(['Inliers - ',num2str(inlierPrePoints.Count), ' points']);
    
    % Create plots y guardar
    % Inliers
    [filepath,name,ext] = fileparts(imds.Files(currFrameIdx));
    conn = findConnection(vSetKeyFrames,lastKeyFrameIdx,currFrameIdx);
    inlierPrePoints = [vSetKeyFrames.Views.Points{lastKeyFrameIdx}.Location(conn.Matches{1}(:,1),1), vSetKeyFrames.Views.Points{lastKeyFrameIdx}.Location(conn.Matches{1}(:,1),2)];
    inlierCurrPoints = [vSetKeyFrames.Views.Points{currFrameIdx}.Location(conn.Matches{1}(:,1),1), vSetKeyFrames.Views.Points{currFrameIdx}.Location(conn.Matches{1}(:,1),2)];

    f1 = figure;
      showMatchedFeatures(currI_corr{currFrameIdx}, currI_corr{lastKeyFrameIdx}, inlierPrePoints, inlierCurrPoints, 'blend');
    title(['Inliers - ',num2str(length(conn.Matches{1})), ' points']);
    saveas(f1,[pathOutputs,'/Inliers_montage',name,'.png']) ;
    close(f1);

    f1 = figure;
    showMatchedFeatures(currI_corr{currFrameIdx}, currI_corr{lastKeyFrameIdx}, inlierPrePoints, inlierCurrPoints, 'blend');
    title(['Inliers - ',num2str(length(conn.Matches{1})), ' points']);
    saveas(f1,[pathOutputs,'/Inliers_blend',name,'.png']) ;
    close(f1);

    % fin loop
    % actualziar ID e indices
    lastKeyFrameId  = currKeyFrameId;
    lastKeyFrameIdx = currFrameIdx;
    addedFramesIdx  = [addedFramesIdx; currFrameIdx]; 
    currFrameIdx    = currFrameIdx + 1;
    
    disp(['Imagen iteracion ', num2str(currFrameIdx),'... DONE'])    
end
%% test
pcshow(mapPointSet.WorldPoints,'VerticalAxis','y','VerticalAxisDir','down','MarkerSize',45)
plotCamera(vSetKeyFrames.Views)
plotCameraPose(vSetKeyFrames.Views(1,:).AbsolutePose.R,vSetKeyFrames.Views(1,:).AbsolutePose.Translation, 1.5) 

%% PLOTS FINAL
plot_info = 1; 
if plot_info
    % Plot Trajectory and 3D Points
    h = figure;
    subplot(2,2,[1,3])
    for i=1:(vSetKeyFrames.NumViews)
        plotCameraPose(vSetKeyFrames.Views(i,:).AbsolutePose.R,vSetKeyFrames.Views(i,:).AbsolutePose.Translation', 0.05);hold on
    end
    title('SLAM 3d points + camera pose');
    hold on;
    scatter3(mapPointSet.WorldPoints(:,1), mapPointSet.WorldPoints(:,2), mapPointSet.WorldPoints(:,3),1, 'b');
    xlim([-1,1])
    ylim([-1,1])
    zlim([-1,1])
    % front view
    subplot(222)
    plot(mapPointSet.WorldPoints(:,1), mapPointSet.WorldPoints(:,2),'.b');
    for i=1:(vSetKeyFrames.NumViews)
        plotCameraPoseXY(vSetKeyFrames.Views(i,:).AbsolutePose.R,vSetKeyFrames.Views(i,:).AbsolutePose.Translation', 0.05);hold on
    end
    xlim([-1,1])
    ylim([-1,1])
    ylabel('Y')
    xlabel('X')
    title ('Front view')

    subplot(224)
    plot(mapPointSet.WorldPoints(:,1), mapPointSet.WorldPoints(:,3),'.b');
    for i=1:(vSetKeyFrames.NumViews)
        plotCameraPoseXZ(vSetKeyFrames.Views(i,:).AbsolutePose.R,vSetKeyFrames.Views(i,:).AbsolutePose.Translation', 0.05);hold on
    end
    xlim([-1,1])
    ylim([-1,1])
    ylabel('Z')
    xlabel('X')
    title ('Top view')
    %% TODO 
    % plot numero puntos ORB detectadas
    % plot numero matches
    % plot numero inliers
    % vs numero de frame/iteracion
end