function [window,targetLoc, bboxes, R] = ...
    im_roi_crop(im, target, bboxes, crop_mode, crop_size, padding, minIn, maxIn, mean_rgb)
% window = im_crop(im, bboxes, crop_mode, crop_size, padding, image_mean)
%   Crops a window specified by bboxes (in [x1 y1 x2 y2] order) out of im.
%
%   crop_mode can be either 'warp' or 'square'
%   crop_size determines the size of the output window: crop_size x crop_size
%   padding is the amount of padding to include at the target scale
%   image_mean to subtract from the cropped window
%
%   N.B. this should be as identical as possible to the cropping
%   implementation in Caffe's WindowDataLayer, which is used while
%   fine-tuning.

% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
%
% This file is part of the R-CNN code and is available
% under the terms of the Simplified BSD License provided in
% LICENSE. Please retain this notice and LICENSE if you use
% this file (or any portion of it) in your project.
% ---------------------------------------------------------
%
% Modified by Hyeonseob Nam, 2015
%

if nargin<6, mean_rgb = []; end

use_square = false;
if strcmp(crop_mode, 'square')
    use_square = true;
end

% defaults if padding is 0
pad_w = 0;
pad_h = 0;
crop_width = crop_size*ones(length(bboxes),1);
crop_height = crop_size*ones(length(bboxes),1);
if padding > 0 || use_square
    %figure(1); showboxesc(im/256, bboxes, 'b', '-');
    scale = crop_size/(crop_size - padding*2);
%    if use_square
%        % make the box a tight square
%        if half_height > half_width
%            half_width = half_height;
%        else
%            half_height = half_width;
%        end
%    end
    bboxes = round(bboxes*scale);
    unclipped_height = bboxes(:,4)-bboxes(:,2)+1;
    unclipped_width = bboxes(:,3)-bboxes(:,1)+1;
    %figure(1); showboxesc([], bboxes, 'r', '-');
    pad_x1 = max(0, 1 - bboxes(:,1));
    pad_y1 = max(0, 1 - bboxes(:,2));
    % clipped bboxes
    bboxes(:,1) = max(1, bboxes(:,1));
    bboxes(:,2) = max(1, bboxes(:,2));
    bboxes(:,3) = min(size(im,2), bboxes(:,3));
    bboxes(:,4) = min(size(im,1), bboxes(:,4));
    clipped_height = bboxes(:,4)-bboxes(:,2)+1;
    clipped_width = bboxes(:,3)-bboxes(:,1)+1;
    scale_x = crop_size./unclipped_width;
    scale_y = crop_size./unclipped_height;
    crop_width = round(clipped_width.*scale_x);
    crop_height = round(clipped_height.*scale_y);
    pad_x1 = round(pad_x1.*scale_x);
    pad_y1 = round(pad_y1.*scale_y);
    
    pad_h = pad_y1;
    pad_w = pad_x1;
    
    crop_height = min( crop_height, crop_size - pad_y1 );
    crop_width = min( crop_width, crop_size - pad_x1 );
end
minLw = min(max(1, bboxes(:,1)));
minLh = min(max(1, bboxes(:,2)));
maxRw = max(min(bboxes(:,3), size(im, 2)));
maxRh = max(min(bboxes(:,4), size(im, 1)));

window = im(minLh:maxRh, minLw:maxRw, :);
bboxes = [ bboxes(:,1) - minLw + 1, bboxes(:, 2) - minLh + 1, bboxes(:, 3) - minLw + 1, bboxes(:,4) - minLh + 1];
targetLoc = [target(1) - minLw + 1, target(2) - minLh + 1, target(3) - minLw + 1, target(4) - minLh + 1];



twidth = maxRw - minLw + 1;
theight = maxRh - minLh + 1;
scale_h = mean(crop_height ./(bboxes(:,4) - bboxes(:,2) + 1));
scale_w = mean(crop_width ./(bboxes(:,3) - bboxes(:,1) + 1));
scale = min(scale_h, scale_w);
scale = max(min( scale, maxIn/max(twidth, theight)), minIn/min(twidth, theight) ); 
input_size = round(min(maxIn, max(theight*scale, twidth*scale)));
tmp = imresize(window, [ input_size, input_size ], 'bilinear', 'antialiasing', false);
scale_h = input_size / theight;
scale_w = input_size / twidth;
tmp = single(tmp);

if isempty(mean_rgb)
%     mean_rgb = mean(mean(tmp));
%     tmp(:,:,1) = tmp(:,:,1)-mean_rgb(1);
%     tmp(:,:,2) = tmp(:,:,2)-mean_rgb(2);
%     tmp(:,:,3) = tmp(:,:,3)-mean_rgb(3);
    tmp = tmp -128;
else
    tmp(:,:,1) = tmp(:,:,1)-mean_rgb(1);
    tmp(:,:,2) = tmp(:,:,2)-mean_rgb(2);
    tmp(:,:,3) = tmp(:,:,3)-mean_rgb(3);
end
window = tmp;
bboxes = [bboxes(:,1)*scale_w, bboxes(:, 2)*scale_h, bboxes(:,3)*scale_w, bboxes(:,4) *scale_h ];
targetLoc = [targetLoc(1)*scale_w, targetLoc(2)*scale_h, targetLoc(3) *scale_w, targetLoc(4) *scale_h ];
R = [minLw, minLh, scale_w, scale_h];
