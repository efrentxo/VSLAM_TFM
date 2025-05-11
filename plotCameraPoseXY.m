function plotCameraPoseXZ(R, t, scale)
    % PLOTCAMERAPOSE - Plots a camera pose using a pyramid model
    % INPUTS:
    %   R     - 3x3 Rotation matrix (camera orientation)
    %   t     - 3x1 Translation vector (camera position)
    %   scale - (Optional) Scale factor for the camera size (default: 1)

    if nargin < 3
        scale = 1; % Default scale if not provided
    end

    % Define pyramid (camera frustum) vertices in camera coordinates
    % Camera at the origin looking in -Z direction
    p0 = [0; 0; 0];                         % Camera center
    p1 = scale * [ 1;  1; 2];  % Top-right
    p2 = scale * [-1;  1; 2];  % Top-left
    p3 = scale * [-1; -1; 2];  % Bottom-left
    p4 = scale * [ 1; -1; 2];  % Bottom-right

    % Transform to world coordinates: X_world = R * X_cam + t
    p0 = R * p0 + t;
    p1 = R * p1 + t;
    p2 = R * p2 + t;
    p3 = R * p3 + t;
    p4 = R * p4 + t;

    % Plot frustum edges
    hold on; grid on;
    
    % Plot pyramid lines (camera frustum)
    plot([p0(1) p1(1)], [p0(2) p1(2)], 'k', 'LineWidth', 2);
    plot([p0(1) p2(1)], [p0(2) p2(2)], 'k', 'LineWidth', 2);
    plot([p0(1) p3(1)], [p0(2) p3(2)], 'k', 'LineWidth', 2);
    plot([p0(1) p4(1)], [p0(2) p4(2)], 'k', 'LineWidth', 2);

    % Connect frustum base
    plot([p1(1) p2(1)], [p1(2) p2(2)], '--k', 'LineWidth', 2);
    plot([p2(1) p3(1)], [p2(2) p3(2)], '--k', 'LineWidth', 2);
    plot([p3(1) p4(1)], [p3(2) p4(2)], '--k', 'LineWidth', 2);
    plot([p4(1) p1(1)], [p4(2) p1(2)], '--k', 'LineWidth', 2);

    % Plot coordinate axes at the camera center
%     quiver3(t(1), t(2), t(3), R(1,1), R(2,1), R(3,1), scale, 'r', 'LineWidth', 2); % X (Red)
%     quiver3(t(1), t(2), t(3), R(1,2), R(2,2), R(3,2), scale, 'g', 'LineWidth', 2); % Y (Green)
%     quiver3(t(1), t(2), t(3), R(1,3), R(2,3), R(3,3), scale, 'b', 'LineWidth', 2); % Z (Blue)
% 
%     % Labels and visualization settings
%     xlabel('X'); ylabel('Y'); zlabel('Z');
%     axis equal;
%     view(3);
end
