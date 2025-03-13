function plotCameraPoseXZ(vSetKeyFrames)
    % PLOTCAMERAPOSE - Plots a camera pose using a pyramid model
    % INPUTS:
    %   vSetKeyFrames

    x = vSetKeyFrames.Views.AbsolutePose.Translation(1,1)
    y = 
    z = 
    plot3(x,y,z, 'r');
end
