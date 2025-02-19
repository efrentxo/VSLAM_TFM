function [features, validPoints, allPoints] = DetectAndExtractFeatures(Irgb, ...
    scaleFactor, numLevels, numPoints, varargin)

if nargin > 4
    intrinsics = varargin{1};
    Irgb  = undistortImage(Irgb, intrinsics);
end

% Detect ORB features
Igray  = im2gray(Irgb);

allPoints = detectORBFeatures(Igray, ScaleFactor=scaleFactor, NumLevels=numLevels);

% Select a subset of features, uniformly distributed throughout the image
points = selectUniform(allPoints, numPoints, size(Igray, 1:2));

% Extract features
[features, validPoints] = extractFeatures(Igray, points);
end