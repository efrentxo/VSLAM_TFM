function plotTrayectoryXZ(vSetKeyFrames,color)
    % PLOTCAMERAPOSE - Plots camera trayectory XZ
    % INPUTS:
    %   vSetKeyFrames
    for i=1:height(vSetKeyFrames.Views)
        x(i) = vSetKeyFrames.Views.AbsolutePose(i,1).Translation(1);
        z(i) = vSetKeyFrames.Views.AbsolutePose(i,1).Translation(3);
    end
    plot(x,z, color);
end