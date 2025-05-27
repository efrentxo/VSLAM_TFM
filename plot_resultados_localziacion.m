function plot_resultados_localziacion(currPose, mapPointSet, pathOutputs, sufix)

%% PLOTS FINAL
steps_points = 10; % porcentaje de puntos a plotear

% Plot Trajectory and 3D Points
% h1 = figure;
% % subplot(2,2,[1,3])
% title('VSLAM 3D points + camera pose');
% hold on;
% scatter3(mapPointSet.WorldPoints(1:steps_points:end,1), mapPointSet.WorldPoints(1:steps_points:end,2), mapPointSet.WorldPoints(1:steps_points:end,3),1, 'b');
% hold on;
% plotTrayectoryXYZ(vSetKeyFrames,'red')
% for i=1:steps_camera:(vSetKeyFrames.NumViews)
%     plotCameraPose(vSetKeyFrames.Views(i,:).AbsolutePose.R,vSetKeyFrames.Views(i,:).AbsolutePose.Translation', 0.05);hold on
% end
% legend('3D points', 'Trayectoria', 'Camara')
% saveas(h1,[pathOutputs,'/Final_VSLAM_3D_mapa_',sufix,'.png']) ;
% saveas(h1,[pathOutputs,'/Final_VSLAM_3D_mapa_',sufix,'.fig']) ;
% close(h1);

% Front view
% subplot(222)
% h2 = figure; 
% plot(mapPointSet.WorldPoints(1:steps_points:end,1), mapPointSet.WorldPoints(1:steps_points:end,2),'.b');
% hold on;
% plotTrayectoryXY(vSetKeyFrames,'red');
% hold on;
% for i=1:steps_camera:(vSetKeyFrames.NumViews)
%     plotCameraPoseXY(1*vSetKeyFrames.Views(i,:).AbsolutePose.R,vSetKeyFrames.Views(i,:).AbsolutePose.Translation', 0.05);hold on
% end
% set(gca, 'YDir','reverse')
% 
% ylabel('Y')
% xlabel('X')
% title ('Front view')
% legend('3D points', 'Trayectoria', 'Camara')
% saveas(h2,[pathOutputs,'/Final_VSLAM_FrontView',sufix,'.png']);
% saveas(h2,[pathOutputs,'/Final_VSLAM_FrontView',sufix,'.fig']);
% close(h2);

% Top view
% subplot(224)
h3 = figure;
plot(mapPointSet.WorldPoints(1:steps_points:end,1), mapPointSet.WorldPoints(1:steps_points:end,3),'.b');
hold on;
% plotTrayectoryXZ(currPose,'red')
hold on;
for i=1:size(currPose,2)
    plotCameraPoseXZ(currPose(i).R,currPose(i).Translation', 0.05);hold on
end

ylabel('Z (m)')
xlabel('X (m)')
title ('Top view')
legend('3D points', 'Camaras')
saveas(h3,[pathOutputs,'/Localizacion_VSLAM_TopView',sufix,'.png']);
saveas(h3,[pathOutputs,'/Localizacion_VSLAM_TopView',sufix,'.fig']);
% close(h3);


