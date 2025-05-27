function plot_resultados_intermedios(vSetKeyFrames,mapPointSet, pathOutputs, sufix)

%% PLOTS FINAL
steps_points = 1; % porcentaje de puntos a plotear
steps_camera = 1; % porcentaje camera pose a plotear

% extraer la ultima camara abs
ind = vSetKeyFrames.NumViews;
Camara_ultima = vSetKeyFrames.Views(ind,:);

% Top view
% subplot(224)
h3 = figure;
plot(mapPointSet.WorldPoints(1:steps_points:end,1), mapPointSet.WorldPoints(1:steps_points:end,3),'.b');
hold on;
plotTrayectoryXZ(vSetKeyFrames,'red')
hold on;
plotCameraPoseXZ(Camara_ultima.AbsolutePose.R,Camara_ultima.AbsolutePose.Translation', 0.05);hold on

ylabel('Z (m)')
xlabel('X (m)')
title ('Top view')
legend('3D points', 'Trayectoria', 'Camara')
saveas(h3,[pathOutputs,'/Tracking_VSLAM_TopView',sufix,'.png']);
saveas(h3,[pathOutputs,'/Tracking_VSLAM_TopView',sufix,'.fig']);
close(h3);


