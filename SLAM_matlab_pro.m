clc; clear; close all;

% mover el path donde esta el fichero actual
file_path = fileparts(matlab.desktop.editor.getActiveFilename);
cd(file_path);

%% Settings
% seleccionar path con la base de dato de imagenes
% imageFolder   = '/home/efren/Escritorio/TFM/datos/Crazyflie_test/';
% imageFolder   = '/home/efren/Escritorio/TFM/datos/Crazyflie_toma2/';
imageFolder   = '/home/efren/Escritorio/TFM/datos/Crazyflie_toma4/';
% imageFolder   = '//home/efren/Escritorio/TFM/datos/Camara_movil_test2//';
% imageFolder   = '//home/efren/Escritorio/TFM/datos/Camara_movil_test5//';

imds          = imageDatastore(imageFolder);
disp(['Imagenes cargadas: ', num2str(numel(imds.Files))])

% Crear subfolder para guardar outputs
currentDateTime = datetime('now', 'Format', 'yyyy-MM-dd_HH-mm-ss');
NewSubFolder= ['Output_',char(currentDateTime)];
if ~exist(fullfile(imageFolder,NewSubFolder))
    mkdir(fullfile(imageFolder,NewSubFolder));
end
pathOutputs = (fullfile(imageFolder,NewSubFolder));

% seleccionar si generar plots para guardar
CrearPlots = 1;
mostrar_figuras = 'off';

% ORB features settings
scaleFactor = 1.2;
numLevels   = 8;
numPoints   = 500;

%% Cargar coeficientes de calibracion de la camara
load('/home/efren/Escritorio/TFM/Calibrar Crazyflie/CrazyFlie_CameraCalibration06022025_3coeff.mat')
% load('/home/efren/PycharmProjects/TFM/Calibrar/CameraCalibration_3coeff_movil.mat');
% cameraParams_3coeff = cameraParams_movil;
% cameraParams = cameraParams_3coeff;

focalLength    = cameraParams.FocalLength;       % in units of pixels
principalPoint = cameraParams.PrincipalPoint;    % in units of pixels
imageSize      = cameraParams.ImageSize;         % in units of pixels    
RadialDistorsion  = [cameraParams.RadialDistortion];
RadialDistorsion  = [0 0 0];                     % not used
TangencialDistorsion = cameraParams.TangentialDistortion;
TangencialDistorsion = [0 0];                    % not used

intrinsics     = cameraIntrinsics(focalLength, principalPoint, imageSize, "RadialDistortion",RadialDistorsion, "TangentialDistortion",TangencialDistorsion);
disp('Camera settings:')
disp([' -- Resolucion: ', num2str(imageSize(1)),'x',num2str(imageSize(2))]);
disp([' -- Focal Length: f1=', num2str(focalLength(1)),' & f2=',num2str(focalLength(2))]);

disp('Camera settings loaded');

%% Analizar primera imagen
% leer la primera imagen
disp ('Analizando primera imagen...')
currFrameIdx = 1;
currI = readimage(imds, currFrameIdx);

[~ ,name,ext] = fileparts(imds.Files(currFrameIdx));
disp (['Imagen - ',num2str(currFrameIdx),'/',num2str(numel(imds.Files)),'- ',name, ext]) 

% Check camera checkings vs image resolution match
RealImageSize      = size(currI,[1 2]);              % in units of pixels
if RealImageSize(2) ~= cameraParams.ImageSize(1) ||  RealImageSize(1)~= cameraParams.ImageSize(2)
    error('ERROR: Los settings de calibracion NO cuadran con las imagenes.');
else
    disp('Los settings de calibracion cuadran con las imagenes.');
end

% Set random seed for reproducibility
rng(0);

% Corregir imagen
currI_corr{currFrameIdx}  = undistortImage(currI, intrinsics);

% Detect and extract ORB features
[preFeatures, prePoints, preAllPoints] = DetectAndExtractFeatures(currI_corr{currFrameIdx}, scaleFactor, numLevels, numPoints, intrinsics); 

% Guardo la primera imagen
firstI       = currI_corr{currFrameIdx}; 

% Aumento indice con el contador de imagenes
currFrameIdx = currFrameIdx + 1;

% Flag para confirmar la inicializacion del mapa
isMapInitialized  = false;

disp ('Analizando primera imagen...DONE')

%% Step - 2 Loop para inicializar mapa 3D
disp ('Step 2 - Inicializacion del mapa 3D.... ')
while ~isMapInitialized && currFrameIdx < numel(imds.Files)
%     disp (['presiona cualquier tecla para continuar...'])
%     pause;
    currI = readimage(imds, currFrameIdx);
    disp (['Step 2 - Imagen - ',num2str(currFrameIdx),'/',num2str(numel(imds.Files)),' - ',name, ext]) 

    % Corregir imagen
    currI_corr{currFrameIdx}  = undistortImage(currI, intrinsics);

    [currFeatures, currPoints] = DetectAndExtractFeatures(currI_corr{currFrameIdx} , scaleFactor, numLevels, numPoints); 

    currFrameIdx = currFrameIdx + 1;

    % Encontrar match entre features
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

    % Obtener Essential matrix aplicando RANSAC
    clear inliersIdx
    [E, inliersIdx] = estimateEssentialMatrix(preMatchedPoints, currMatchedPoints, cameraParams,'MaxNumTrials', 500, 'Confidence', 99.9, 'MaxDistance', 1);
    
    inlierTformIdx = inliersIdx; % delete?
    tform = E;

    % Calcular camera pose - usamos todos los puntos? o solo una parte?
    % por defecto se usan todos los puntos
    inlierPrePoints  = preMatchedPoints(inliersIdx);
    inlierCurrPoints = currMatchedPoints(inliersIdx);
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


    % Camera relative pose between frames
    [relPose, validFraction] = estrelpose(tform, intrinsics, inlierPrePoints(1:2:end), inlierCurrPoints(1:2:end));

    % If not enough inliers are found, move to the next frame
    if validFraction < 0.9 || numel(relPose)>1
        disp (['Imagen - ',num2str(currFrameIdx-1),'...no sufientes inliers']);
        continue
    end

    % Triangulate two views to obtain 3-D map points
    minParallax = 20; % In deg
    [isValid, xyzWorldPoints, inlierTriangulationIdx] = TriangulateTwoFrames(...
        rigidtform3d, relPose, inlierPrePoints, inlierCurrPoints, intrinsics, minParallax);
    
    % Q&D
    isValid = 1;
    if ~isValid
        disp (['Imagen - ',num2str(currFrameIdx-1),'...triangulacion no valida']);
        continue
    end
    % Q&D end

    inlierPrePoints  = inlierPrePoints(inlierTriangulationIdx);
    inlierCurrPoints = inlierCurrPoints(inlierTriangulationIdx);

    % Display Matches & Final Inliers 
    f1 = figure;
    subplot(211)
    showMatchedFeatures(firstI, currI_corr{currFrameIdx-1}, preMatchedPoints, currMatchedPoints, 'montage');
    title(['Inicializar-Matches - ',num2str(currMatchedPoints.Count), ' points']);

    subplot(212);
    showMatchedFeatures(firstI, currI_corr{currFrameIdx-1}, inlierPrePoints, inlierCurrPoints, 'montage');
    title(['Inicializar-Inliers - ',num2str(inlierPrePoints.Count), ' points']);
    saveas(f1,[pathOutputs,'/Inicializar-MatchesInliersPoints.png']) ;
    close(f1);

    % Get the original index of features in the two key frames
    indexPairs = indexPairs(inlierTriangulationIdx,:);
    
%     % Display Final Inliers check
%     figure;
%     showMatchedFeatures(firstI , currI_corr{currFrameIdx-1} , prePoints(indexPairs(:,1)), currPoints(indexPairs(:,2)), 'blend');
%     title(['Inliers - ',num2str(inlierPrePoints.Count), ' points']);

    isMapInitialized = true;

end
disp(['Step 2 - Mapa 3D inicializado con imagenes 1 e imagen ', num2str(currFrameIdx-1)])
disp ('Step 2 - Inicializacion del mapa 3D.... DONE ')

%% Iniciar mapa3d y views 
% Create an empty imageviewset object to store key frames
disp(['Step 2 - Almacenar mapa 3d y camera pose... '])
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

disp(['Step 2 - Almacenar mapa 3d y camera pose... DONE'])
 
disp(['Step 2 - Aplicar bundle adjustment... '])
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

disp(['Step 2 - Aplicar bundle adjustment...DONE '])

%% Step 3 - Tracking
% ViewId of the current key frame
disp ('Step 3 - Leer el resto de imagenes... ')
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

while currFrameIdx <= numel(imds.Files)  
    disp(['Imagen iteracion ', num2str(currFrameIdx),'/',num2str(numel(imds.Files))])    
    currI = readimage(imds, currFrameIdx);
    % Corregir imagen
    currI_corr{currFrameIdx}  = undistortImage(currI, intrinsics);
    
%     figure;
%     imshow(currI_corr{currFrameIdx-1})
    [currFeatures, currPoints] = DetectAndExtractFeatures(currI_corr{currFrameIdx} , scaleFactor, numLevels, 500);
 
    f1 = figure('visible',mostrar_figuras);
    [filepath,name,ext] = fileparts(imds.Files(currFrameIdx));
%     imshow(currI_corr{currFrameIdx-1})
    imshow(currI_corr{currFrameIdx});hold on;
    plot(currPoints,'ShowScale',false, showOrientation=false)
    title(['All ORB - ',num2str(currPoints.Count), ' points - ',name,' - (',num2str(currFrameIdx),'/',num2str(numel(imds.Files)),')'],'Interpreter','None');
    saveas(f1,[pathOutputs,'/All_ORB_points_',name,'.png']) ;
    close(f1);


    % Track the last key frame
    [currPose, mapPointsIdx, featureIdx] = helperTrackLastKeyFrame(...
    mapPointSet, vSetKeyFrames.Views, currFeatures, currPoints, lastKeyFrameId, intrinsics, scaleFactor);

    if isempty(mapPointsIdx) || length(mapPointsIdx)< 6
        disp('no se han encotrado matches con los 3d points')
        currFrameIdx = currFrameIdx + 1;
        continue;
    end
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
    conn = findConnection(vSetKeyFrames, lastKeyFrameIdx, currFrameIdx);
    ind_prev = conn.Matches{1}(:,1);
    ind_curr = conn.Matches{1}(:,2);
    inlierPrePoints  = [vSetKeyFrames.Views.Points{lastKeyFrameIdx}.Location(ind_prev,:)];
    inlierCurrPoints = [vSetKeyFrames.Views.Points{currFrameIdx}.Location(ind_curr,:)];

%     f1 = figure('visible',mostrar_figuras);
%     showMatchedFeatures(currI_corr{lastKeyFrameIdx}, currI_corr{currFrameIdx}, inlierPrePoints, inlierCurrPoints, 'montage');
%     title(['Inliers - ',num2str(length(conn.Matches{1})), ' points - ',name,' -(',num2str(currFrameIdx),'/',num2str(numel(imds.Files)),')'],'Interpreter','None');
%     saveas(f1,[pathOutputs,'/Inliers_montage_',name,'.png']) ;
%     close(f1);

    f1 = figure('visible',mostrar_figuras);
    showMatchedFeatures(currI_corr{lastKeyFrameIdx}, currI_corr{currFrameIdx}, inlierPrePoints, inlierCurrPoints, 'blend');
    title(['Inliers - ',num2str(length(conn.Matches{1})), ' points - ',name,' -(',num2str(currFrameIdx),'/',num2str(numel(imds.Files)),')'],'Interpreter','None');
    saveas(f1,[pathOutputs,'/Inliers_blend_',name,'.png']) ;
    close(f1);

    % fin loop
    % actualziar ID e indices
    lastKeyFrameId  = currKeyFrameId;
    lastKeyFrameIdx = currFrameIdx;
    addedFramesIdx  = [addedFramesIdx; currFrameIdx]; 
    currFrameIdx    = currFrameIdx + 1;
    
    disp(['Imagen iteracion ', num2str(currFrameIdx-1),'... DONE'])      

end
disp('Step 3 - Leer el resto de imagenes... DONE')

%% Step 4 - Check Loop Closure
step4 = 0;
disp('Step 4 - Check loop closure... ')
if step4

    % Detectar loop closure
    % cada 20 keyf
    % mathing features - propueta sencilla pero pesada compitacopalmente
    % hacer el match con el current frama y los features delos frames
    % pasados
    for i = 1:vSetKeyFrames.NumViews
        indexPairs = matchFeatures(binaryFeatures(vSetKeyFrames.Views.Features{i,1}), currFeatures, 'MaxRatio', 0.7);
        LoopClosure_matches(i) = length(indexPairs);
        if length(indexPairs) > 20  % Set threshold for loop closure
            loopClosureDetected = true;
            matchedFrameIdx = i;
            disp(['Loop closure detected between current frame...', num2str(matchedFrameIdx)])
        end
    end
    
    % Aplicar loop closure
    LoopClosureDetected = 1;
    if LoopClosureDetected 
        % Q&D manual loop closure para chequear como queda el mapa3d
        id1 = 1;
        id2 = 418;
        [filepath1,name1,ext1] = fileparts(imds.Files(id1));
        [filepath2,name2,ext2] = fileparts(imds.Files(id2));
    
        % extraer los indicies de los puntos 3d de la imagen actual, y los indices de los
        % feaures correspondientes 
        [index3d1, index2d1] = findWorldPointsInView(mapPointSet, id1);
        allFeatures1   = vSetKeyFrames.Views.Features{id1};
        validFeatures1  = allFeatures1(index2d1, :);
       
        [index3d2, index2d2] = findWorldPointsInView(mapPointSet, id2);
        allFeatures2   = vSetKeyFrames.Views.Features{id2};
        validFeatures2 = allFeatures2(index2d2, :);
    
        indexPairs = matchFeatures(binaryFeatures(validFeatures1), binaryFeatures(validFeatures2), ...
            'Unique', true, 'MaxRatio', 0.9, 'MatchThreshold', 40);
    
%         % Check
%         [FeaId1, PointsId1] = DetectAndExtractFeatures(currI_corr{1} , scaleFactor, numLevels, 500);
%         [FeaId2, PointsId2] = DetectAndExtractFeatures(currI_corr{id2} , scaleFactor, numLevels, 500);
%         indexPairs = matchFeatures(FeaId1, FeaId2, ...
%             'Unique', true, 'MaxRatio', 0.9, 'MatchThreshold', 40);
%         clear inliersIdx
%         [E, inliersIdx] = estimateEssentialMatrix(PointsId1(indexPairs(:,1)), PointsId2(indexPairs(:,2)), cameraParams,'MaxNumTrials', 500, 'Confidence', 99.9, 'MaxDistance', 1);
%         
%         Inliers1 = PointsId1(indexPairs(:,1));
%         Inliers1 = Inliers1(inliersIdx);
%         Inliers2 = PointsId2(indexPairs(:,2));
%         Inliers2 = Inliers2(inliersIdx);
%         showMatchedFeatures(currI_corr{id1}, currI_corr{id2}, Inliers1, Inliers2, 'montage');
%     
%         % chequeo la info guardada, me da lo mismo
%         sum(sum(FeaId1.Features - vSetKeyFrames.Views.Features{loopCandidates(1)}))
%         sum(sum(FeaId2.Features - vSetKeyFrames.Views.Features{418}))
%         sum(sum(FeaId2.Features - currFeatures.Features))
%         % end check
    
        % mostrar imagenes
        figure; 
        imshow(currI_corr{id1});
        figure; 
        imshow(currI_corr{id2});
    
        worldPoints1 = mapPointSet.WorldPoints(index3d1(indexPairs(:, 1)), :);
        worldPoints2 = mapPointSet.WorldPoints(index3d2(indexPairs(:, 2)), :);
    
        tform1 = pose2extr(vSetKeyFrames.Views.AbsolutePose(id1));
        tform2 = pose2extr(vSetKeyFrames.Views.AbsolutePose(id2));
    
        worldPoints1InCamera1 = transformPointsForward(tform1, worldPoints1) ;
        worldPoints2InCamera2 = transformPointsForward(tform2, worldPoints2) ;
    
        [tform, inlierIndex] = estgeotform3d(...
            worldPoints1InCamera1, worldPoints2InCamera2, 'similarity', 'MaxDistance', 0.1);
    
        f1 = figure('visible','on');
        points1_location = vSetKeyFrames.Views.Points{id1}.Location(index2d1(indexPairs(:, 1)),:);     
        points2_location = vSetKeyFrames.Views.Points{id2}.Location(index2d2(indexPairs(:, 2)),:);   
        points1_location_inliers  = points1_location(inlierIndex,:);     
        points2_location_inliers  = points2_location(inlierIndex,:); 
    
        subplot(211);
        showMatchedFeatures(currI_corr{id1}, currI_corr{id2}, points1_location, points2_location, 'montage');
        title({['Loop Closure - ',num2str(length(points1_location)),' match points'],...+
               [name1,' - ' name2]},...
               'Interpreter','None');
        subplot(212);
        showMatchedFeatures(currI_corr{id1}, currI_corr{id2}, points1_location_inliers, points2_location_inliers, 'montage');
        title({['Loop Closure - ',num2str(length(points1_location_inliers)),' inlier points'],...+
               [name1,' - ' name2]},...
               'Interpreter','None');
        saveas(f1,[pathOutputs,'/LoopClosure_blend.png']) ;
        close(f1);
    
        % Add connection between the current key frame and the loop key frame
        matches = uint32([index2d1(indexPairs(inlierIndex, 1)), index2d2(indexPairs(inlierIndex, 2))]);
        
        vSetKeyFrames_LoopClosure = addConnection(vSetKeyFrames, id1, id2, tform, 'Matches', matches);
%         vSetKeyFrames = deleteConnection(vSetKeyFrames,id1,id2);
        disp(['Loop closure añadido entre imagen: ', num2str(id1), ' y ', num2str(id2)]);
    
        % Fuse co-visible map points
        matchedIndex3d1 = index3d1(indexPairs(inlierIndex, 1));
        matchedIndex3d2 = index3d2(indexPairs(inlierIndex, 2));
        mapPoints_LoopClosure= updateWorldPoints(mapPointSet, matchedIndex3d2, mapPointSet.WorldPoints(matchedIndex3d1, :));
    
        if length(matches) > 20
            isLoopClosed = 1;
        else
            isLoopClosed = 0;
        end
    end

    if isLoopClosed
        % Optimize the poses
        minNumMatches      = 20;
        vSetKeyFrames_LoopClosure_Opt = optimizePoses(vSetKeyFrames_LoopClosure, minNumMatches, Tolerance=1e-16);
    
        % Update map points after optimizing the poses
        mapPoints_LoopClosure_Opt = helperUpdateGlobalMap(mapPoints_LoopClosure, vSetKeyFrames_LoopClosure, vSetKeyFrames_LoopClosure_Opt);
    
    end    

else
    disp('Step 4 - skip................')
end
disp('Step 4 - Check loop closure... DONE')

%% test
% TODO 
% añadir en las imaegnes el nombre del fichero
% anadir que imagenes respecto al total de imagenes existentes
% añadir todos los puntos ORB , 
% otro plot con los puntos que existen en el 3d
% otro plot con los datos creados por triangulacion


%% guardar datos
% preguntar si guardar


%% PLOTS FINAL
plot_resultados(vSetKeyFrames, mapPointSet)
plot_resultados(vSetKeyFrames_LoopClosure_Opt, mapPoints_LoopClosure_Opt)

figure;
plotTrayectoryXYZ(vSetKeyFrames)
hold on;
plotTrayectoryXYZ(vSetKeyFrames,'b')

plot_info = 1; 
steps_points = 5; % porcentaje de puntos a plotear
steps_camera = 5; % porcentaje camera pose a plotear

if plot_info
    % Plot Trajectory and 3D Points
    h = figure;
    subplot(2,2,[1,3])
    for i=1:steps_camera:(vSetKeyFrames.NumViews)
%         plotCameraPose(vSetKeyFrames.Views(i,:).AbsolutePose.R,vSetKeyFrames.Views(i,:).AbsolutePose.Translation', 0.05);hold on
        plotCameraPose(vSetKeyFramesOptim.Views(i,:).AbsolutePose.R,vSetKeyFramesOptim.Views(i,:).AbsolutePose.Translation', 0.05);hold on
    end
    title('SLAM 3d points + camera pose');
    hold on;
    scatter3(mapPointSet.WorldPoints(1:steps_points:end,1), mapPointSet.WorldPoints(1:steps_points:end,2), mapPointSet.WorldPoints(1:steps_points:end,3),1, 'b');
%     xlim([-1,1])
%     ylim([-1,1])
%     zlim([-1,1])

    % Front view
    subplot(222)
    plot(mapPointSet.WorldPoints(:,1), mapPointSet.WorldPoints(:,2),'.b');
    for i=1:(vSetKeyFrames.NumViews)
        plotCameraPoseXY(1*vSetKeyFrames.Views(i,:).AbsolutePose.R,vSetKeyFrames.Views(i,:).AbsolutePose.Translation', 0.05);hold on
    end
    set(gca, 'YDir','reverse')
%     xlim([-1,1])
%     ylim([-1,1])
    ylabel('Y')
    xlabel('X')
    title ('Front view')
    legend ('3D points')

    % Top view
    subplot(224)
    plot(mapPointSet.WorldPoints(:,1), mapPointSet.WorldPoints(:,3),'.b');
    for i=1:(vSetKeyFrames.NumViews)
        plotCameraPoseXZ(vSetKeyFrames.Views(i,:).AbsolutePose.R,vSetKeyFrames.Views(i,:).AbsolutePose.Translation', 0.05);hold on
    end
%     xlim([-1,1])
%     ylim([-1,1])
    ylabel('Z')
    xlabel('X')
    title ('Top view')

end

%%
plot_stats = 1;
if plot_stats
    % TODO 
    % plot numero puntos ORB detectadas por frame DONE
    % plot numero matches con frame anterior DONE
    % plot numero inliers con frame anteior
    % plot numero inliers con el resto de frames anteriores
    % plot numero inliers con puntos ya existentes en 3d DONE
    % vs numero de frame/iteracion
    f2 = figure;
    clear Matches_points_prev Matches_views
    % preparar datos a plotear
    for i=1:vSetKeyFrames.NumViews
        % puntos ORB por cada imagen
        ORB_points_per_view(i) = vSetKeyFrames.Views.Points{i,1}.Count;

        % conection con puntos 3d por frame
        Points3D_per_frame(i) = length(findWorldPointsInView(mapPointSet,i));

        % connection between 2 frames/views
        if i < vSetKeyFrames.NumViews
            % con el frame anterior
            conn = findConnection(vSetKeyFrames,i,i+1);
            Matches_points_prev(i) = length(conn.Matches{1,1});
            
            % con hasta 2 frames consecutivos
            aux = connectedViews(vSetKeyFrames,i,MinNumMatches=10,MaxDistance=2);
            Matches_views(i) = height(aux);
        else
            Matches_points_prev = [0, Matches_points_prev];
            Matches_views = [0, Matches_views];
        end
    end

    % hacer plots
    plot(1:vSetKeyFrames.NumViews,ORB_points_per_view);
    hold on;
    plot(1:vSetKeyFrames.NumViews,Matches_points_prev);
    plot(1:vSetKeyFrames.NumViews,Matches_views);
    plot(1:vSetKeyFrames.NumViews,Points3D_per_frame);
    grid on
    ylim([-10,600])
    xlabel('Imagenes')
    ylabel('Nr')
    legend('ORB puntos por imagen','Matches points','Matches views','3D points por imagen')
            
end