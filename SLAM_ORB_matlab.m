imageFolder   = ['/home/efren/Escritorio/TFM/datos/test2/'];
% imageFolder   = ['/home/efren/Escritorio/TFM/datos/test1/'];
imageFolder   = ['/home/efren/Escritorio/TFM/datos/test1/'];
imds          = imageDatastore(imageFolder);
disp(['Imagenes cargadas: ', num2str(numel(imds.Files))])

% bag = bagOfFeaturesDBoW(imds);

%%
% Inspect the first image
currFrameIdx = 1;
currI = readimage(imds, currFrameIdx);
himage = imshow(currI );

% load('/home/efren/Escritorio/TFM/Calibrar Crazyflie/CrazyFlie_CameraCalibration06022025_3coeff.mat')
load('/home/efren/PycharmProjects/TFM/Calibrar/CameraCalibration_3coeff_movil.mat');


focalLength    = cameraParams_3coeff.FocalLength;    % in units of pixels
principalPoint = cameraParams_3coeff.PrincipalPoint;    % in units of pixels
imageSize      = size(currI,[1 2]);  % in units of pixels
RadialDistorsion  = [cameraParams_3coeff.RadialDistortion];
RadialDistorsion  = [0 0 0];
TangencialDistorsion = cameraParams_3coeff.TangentialDistortion;
intrinsics     = cameraIntrinsics(focalLength, principalPoint, imageSize, "RadialDistortion",RadialDistorsion, "TangentialDistortion",TangencialDistorsion);

% Set random seed for reproducibility
rng(0);

% Detect and extract ORB features
scaleFactor = 1.2;
numLevels   = 8;
numPoints   = 500;
[preFeatures, prePoints, preAllPoints] = DetectAndExtractFeatures(currI, scaleFactor, numLevels, numPoints, intrinsics); 


% plot iteracion 0
figure
subplot(121)
imshow(currI)
hold on
plot(prePoints,'ShowScale',false)
title ('filterpoints')
subplot(122)
imshow(currI)
hold on
plot(preAllPoints,'ShowScale',false)
title ('allpoints')

currFrameIdx = currFrameIdx + 1;
firstI       = currI; % Preserve the first frame 

isMapInitialized  = false;

% Map initialization loop
while ~isMapInitialized && currFrameIdx < numel(imds.Files)
    currI = readimage(imds, currFrameIdx);

    [currFeatures, currPoints] = DetectAndExtractFeatures(currI, scaleFactor, numLevels, numPoints, intrinsics); 
    
    figure  
    imshow(currI)
    hold on
    plot(currPoints,'ShowScale',false)
     
    currFrameIdx = currFrameIdx + 1;

    % Find putative feature matches
    indexPairs = matchFeatures(preFeatures, currFeatures, Unique=true, ...
        MaxRatio=0.9, MatchThreshold=40);

    % If not enough matches are found, check the next frame
    minMatches = 100;
    if size(indexPairs, 1) < minMatches
        continue
    end

    preMatchedPoints  = prePoints(indexPairs(:,1),:);
    currMatchedPoints = currPoints(indexPairs(:,2),:);

    figure; 
    showMatchedFeatures(firstI,currI,preMatchedPoints,currMatchedPoints,"blend");
    figure; ax = axes;
    showMatchedFeatures(firstI,currI,preMatchedPoints,currMatchedPoints,"montag");
    title(ax,"Candidate point matches");
    legend(ax,"Matched points 1","Matched points 2");

    % Filtro RANSAC
     % Estimate motion using Essential Matrix
    [E, inlierIdx] = estimateEssentialMatrix(preMatchedPoints.Location, currMatchedPoints.Location, intrinsics);
    % Recover relative camera pose
    [relativePose, inlierPoints1, inlierPoints2] = relativeCameraPose(E, intrinsics, preMatchedPoints, currMatchedPoints);
    
    % Compute homography and evaluate reconstruction
    [tformH, scoreH, inliersIdxH] = helperComputeHomography(preMatchedPoints, currMatchedPoints);

    % Compute fundamental matrix and evaluate reconstruction
    [tformF, scoreF, inliersIdxF] = helperComputeFundamentalMatrix(preMatchedPoints, currMatchedPoints, intrinsics);

    % Select the model based on a heuristic
    ratio = scoreH/(scoreH + scoreF);
    ratioThreshold = 0.45;
    if ratio > ratioThreshold
        inlierTformIdx = inliersIdxH;
        tform          = tformH;
    else
        inlierTformIdx = inliersIdxF;
        tform          = tformF;
    end

    % Computes the camera location up to scale. Use half of the points to reduce computation
    inlierPrePoints  = preMatchedPoints(inlierTformIdx);
    inlierCurrPoints = currMatchedPoints(inlierTformIdx);
    [relPose, validFraction] = estrelpose(tform, intrinsics, inlierPrePoints(1:2:end), inlierCurrPoints(1:2:end));

    % If not enough inliers are found, move to the next frame
    if validFraction < 0.9 || numel(relPose)>1
        continue
    end

    % Triangulate two views to obtain 3-D map points
    minParallax = 1; % In degrees
    [isValid, xyzWorldPoints, inlierTriangulationIdx] = helperTriangulateTwoFrames(...
        rigidtform3d, relPose, inlierPrePoints, inlierCurrPoints, intrinsics, minParallax);

    if ~isValid
        continue
    end

    % Get the original index of features in the two key frames
    indexPairs = indexPairs(inlierTformIdx(inlierTriangulationIdx),:);

    isMapInitialized = true;

    disp(['Map initialized with frame 1 and frame ', num2str(currFrameIdx-1)])
end % End of map initialization loop


if isMapInitialized
    close(himage.Parent.Parent); % Close the previous figure
    % Show matched features
    hfeature = showMatchedFeatures(firstI, currI, prePoints(indexPairs(:,1)), currPoints(indexPairs(:, 2)), "Montage");
else
    error('Unable to initialize the map.')
end


% Create an empty imageviewset object to store key frames
vSetKeyFrames = imageviewset;

% Create an empty worldpointset object to store 3-D map points
mapPointSet   = worldpointset;

% Add the first key frame. Place the camera associated with the first 
% key frame at the origin, oriented along the Z-axis
preViewId     = 1;
vSetKeyFrames = addView(vSetKeyFrames, preViewId, rigidtform3d, Points=prePoints, Features=preFeatures.Features);

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

%%
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

%%

% Visualize initial map points and camera trajectory
mapPlot       = helperVisualizeMotionAndStructure(vSetKeyFrames, mapPointSet);

% Show legend
showLegend(mapPlot);

% Main loop
isLastFrameKeyFrame = true;

% Create and initialize the KLT tracker
tracker = vision.PointTracker(MaxBidirectionalError = 5);
initialize(tracker, currPoints.Location(indexPairs(:,2), :), currI);

while currFrameIdx < numel(imds.Files)  
    currI = readimage(imds, currFrameIdx);
    [currFeatures, currPoints] = DetectAndExtractFeatures(currI, scaleFactor, numLevels, numPoints, intrinsics);

    % Track the last key frame
    [currPose, mapPointsIdx, featureIdx] = helperTrackLastKeyFrameKLT(tracker, currI, mapPointSet, ...
        vSetKeyFrames.Views, currFeatures, currPoints, lastKeyFrameId, intrinsics);

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
    numSkipFrames     = 20;
    numPointsKeyFrame = 90;
    [localKeyFrameIds, currPose, mapPointsIdx, featureIdx, isKeyFrame] = ...
        helperTrackLocalMap(mapPointSet, vSetKeyFrames, mapPointsIdx, ...
        featureIdx, currPose, currFeatures, currPoints, intrinsics, scaleFactor, ...
        isLastFrameKeyFrame, lastKeyFrameIdx, currFrameIdx, numSkipFrames, numPointsKeyFrame);

    % Visualize matched features
    updatePlot(featurePlot, currI, currPoints(featureIdx));

    if ~isKeyFrame
        currFrameIdx        = currFrameIdx + 1;
        isLastFrameKeyFrame = false;
        continue
    else
        currKeyFrameId      = currKeyFrameId + 1;
        isLastFrameKeyFrame = true;
    end
 
%% TO DO


end