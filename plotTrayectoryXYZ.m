function plotTrayectoryXYZ(vSetKeyFrames,color)
    % PLOTCAMERAPOSE - Plots a camera pose using a pyramid model
    % INPUTS:
    %   vSetKeyFrames
    for i=1:height(vSetKeyFrames.Views)
        x(i) = vSetKeyFrames.Views.AbsolutePose(i,1).Translation(1);
        y(i) = vSetKeyFrames.Views.AbsolutePose(i,1).Translation(2);
        z(i) = vSetKeyFrames.Views.AbsolutePose(i,1).Translation(3);
    end
    plot3(x,y,z, color);
end
