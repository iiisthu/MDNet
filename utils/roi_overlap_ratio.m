function r = roi_overlap_ratio(rect1, rect2)
% OVERLAP_RATIO
% Compute the overlap ratio between two rectangles
%
% Hyeonseob Nam, 2015
% 
rect1 = [rect1(:,1:2) rect1(:,3:4) - rect1(:,1:2) + 1];
rect2 = [rect2(:,1:2) rect2(:,3:4) - rect2(:,1:2) + 1];
inter_area = rectint(rect1,rect2);
union_area = rect1(:,3).*rect1(:,4) + rect2(:,3).*rect2(:,4) - inter_area;

r = inter_area./union_area;
end
