function plotTrayectoryXY(vSetKeyFrames,color)
    % PLOTCAMERAPOSE - Plots camera trayectory XY 
    % INPUTS:
    %   vSetKeyFrames
    for i=1:height(vSetKeyFrames.Views)
        x(i) = vSetKeyFrames.Views.AbsolutePose(i,1).Translation(1);
        y(i) = vSetKeyFrames.Views.AbsolutePose(i,1).Translation(2);  
    end
    plot(x,y, color);
end