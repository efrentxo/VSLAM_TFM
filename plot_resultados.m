function plot_resultados(vSetKeyFrames,mapPointSet)

%% PLOTS FINAL
steps_points = 1; % porcentaje de puntos a plotear
steps_camera = 1; % porcentaje camera pose a plotear

% Plot Trajectory and 3D Points
h = figure;
subplot(2,2,[1,3])
for i=1:steps_camera:(vSetKeyFrames.NumViews)
    plotCameraPose(vSetKeyFrames.Views(i,:).AbsolutePose.R,vSetKeyFrames.Views(i,:).AbsolutePose.Translation', 0.05);hold on
end
title('SLAM 3d points + camera pose');
hold on;
scatter3(mapPointSet.WorldPoints(1:steps_points:end,1), mapPointSet.WorldPoints(1:steps_points:end,2), mapPointSet.WorldPoints(1:steps_points:end,3),1, 'b');

% Front view
subplot(222)
plot(mapPointSet.WorldPoints(:,1), mapPointSet.WorldPoints(:,2),'.b');
for i=1:(vSetKeyFrames.NumViews)
    plotCameraPoseXY(1*vSetKeyFrames.Views(i,:).AbsolutePose.R,vSetKeyFrames.Views(i,:).AbsolutePose.Translation', 0.05);hold on
end
set(gca, 'YDir','reverse')

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

ylabel('Z')
xlabel('X')
title ('Top view')

