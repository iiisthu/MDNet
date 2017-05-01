function [imo, rois, labels, targets] = get_roi_batch(opts, imdb, k, batch, varargin)
% GET_BATCH
% Load, preprocess, and pack images for CNN evaluation
%
% Modified from cnn_imagenet_get_batch() in the MatConvNet library.
% Hyeonseob Nam, 2015
%
%  
%boxes = [pos_idx, pos_boxes; neg_idx, neg_boxes];
opts.numFetchThreads     = 1 ;
opts.prefetch = false;
opts.batch_pos        = floor(opts.batch_pos/numel(batch));
opts.batch_neg        = floor(opts.batch_neg/numel(batch));
opts = vl_argparse(opts, varargin);
images = {imdb.images.name{k}{batch}}; 
gtboxes = {imdb.boxes.gtbox{k}{batch}};
im = cell(1, numel(images)) ;
% fetch is true if images is a list of filenames (instead of
% a cell array of images)
fetch = ischar(imdb.images.name{k}{1}) ;

% prefetch is used to load images in a separate thread
prefetch = fetch & opts.prefetch ;

im = cell(1, numel(images)) ;
if opts.numFetchThreads > 0
    if prefetch
        vl_imreadjpeg(images, 'numThreads', opts.numFetchThreads, 'prefetch');
        imo = [];
        rois =[];
        labels = [];
        targets = [];
      
        return ;
    end
    if fetch
        ims = vl_imreadjpeg(images,'numThreads', opts.numFetchThreads) ;
    end
end
if ~fetch
    ims = images ;
end



maxW = 0;
maxH = 0;


pboxes   = cell(1,numel(batch));
plabels  = cell(1,numel(batch));
ptargets  = cell(1,numel(batch));
% get fg and bg rois
for b=1:numel(batch)
  pbox   = imdb.boxes.pbox{k}{batch(b)};
  plabel = imdb.boxes.plabel{k}{batch(b)};
  ptarget = imdb.boxes.ptarget{k}{batch(b)};

  if size(pbox,2)~=4
    error('wrong box size');
  end
 % get pos boxes
  pos = find((plabel ~= opts.bgLabel)) ;
  npos = numel(pos);
  % get neg boxes
  neg = find((plabel==opts.bgLabel)) ;
  nneg = numel(neg);

    bbox = [];
    label = [];
    target = [];
    nBpos = min(npos,opts.batch_pos);
    nBneg = min(nneg,opts.batch_neg);

    if npos>0
      r = randperm(npos);
      p = pos(r(1:nBpos));
      bbox = pbox(p,:);
      label = plabel(p);
      target = ptarget(p,:);
    end
    if nneg>0
      r = randperm(nneg);

      n = neg(r(1:nBneg));
      bbox = [bbox ; pbox(n,:)];
      label = [label ; plabel(n)];
      target = [target; ptarget(n,:)];
    end
  
  pboxes{b} = bbox;
  plabels{b} = label;
  ptargets{b} = target;
end

labels = vertcat(plabels{:});
targets = vertcat(ptargets{:});
% chose ratio
target = gtboxes{1};
center = [(target(1) + target(3))/2, (target(2) + target(4))/2,(target(3) - target(1) + 1),...
target(4) - target(2) + 1];
crop_range = (opts.trans_range + opts.scale_factor^opts.scale_range);
minLw = round(max(1,  center(1) - center(3)*crop_range));
minLh = round(max(1,  center(2) - center(4)*crop_range));
maxRw = round(min(center(1) + center(3)*crop_range, size(ims{1}, 2)));
maxRh = round(min(center(2) + center(4)*crop_range, size(ims{1}, 1)));
ratio = (maxRw - minLw + 1)/(maxRh - minLh + 1);
aspect_ratio = [1];
[~, as] = min(abs(ratio - aspect_ratio));
as_chosen = aspect_ratio(as);
%fprintf('actual ratio: %.2f, chosen ratio: %.2f', ratio, as_chosen);
% rescale images and rois
rois = [];
imre = cell(1,numel(batch));
for b=1:numel(batch)
  imSize = size(ims{b});

  h = imSize(1);
  w = imSize(2);

%  factor = max(opts.scale(1)/h,opts.scale(1)/w);

%  if any([h*factor,w*factor]>opts.maxScale)
%    factor = min(opts.maxScale/h,opts.maxScale/w);
%  end
%  if abs(factor-1)>1e-3
%    imre{b} = imresize(ims{b},factor,'Method',opts.interpolation);
%  else
%    imre{b} = ims{b};
%  end

%  if imdb.boxes.flip(batch(b))
%    im = imre{b};
%    imre{b} = im(:,end:-1:1,:);
%  end

%  imreSize = size(imre{b});

%  maxH = max(imreSize(1),maxH);
%  maxW = max(imreSize(2),maxW);

  % adapt bounding boxes into new coord
  bbox = pboxes{b};
  plabel = plabels{b};
  if opts.visualize
     figure(1);
     imshow(uint8(ims{b}));
     hold on;
     for i=1:size(bbox,1)
        if plabel(i) == 2
        rectangle('Position',round([bbox(i,1),bbox(i,2), bbox(i,3) - bbox(i,1) + 1, bbox(i,4) - bbox(i,2) + 1]));
        hold on;
        end
     end
     hold off;
  end 

  [imre{b}, targetLoc, bboxes, R] = im_roi_crop(ims{b}, gtboxes{b}, bbox, opts.crop_mode, opts.crop_size, opts.crop_padding, as_chosen, crop_range, []);
   if opts.visualize
%     figure(1);
%     imshow(uint8(ims{b}));
%     bb = [ targetLoc(1:2)./R(3:4) + R(1:2), targetLoc(3:4)./R(3:4) + R(1:2)];
%     rectangle('Position', round(bb));
     
     figure(2);
     imshow(uint8(imre{b}+128));
     hold on;
     l=1;
     for i=1:size(bboxes,1)
         if plabel(i) == 2
            l=i;
            rectangle('Position',round([bboxes(i,1),bboxes(i,2), bboxes(i,3) - bboxes(i,1) + 1, bboxes(i,4) - bboxes(i,2) + 1]),'EdgeColor','r');
         else
            rectangle('Position',round([bboxes(i,1),bboxes(i,2), bboxes(i,3) - bboxes(i,1) + 1, bboxes(i,4) - bboxes(i,2) + 1]),'EdgeColor','b');
         end
            hold on;
     end
     fprintf('Input size: [%d,%d] [%d, %d]\n', size(imre{b},1), size(imre{b},2), round(bboxes(l, 3) - bboxes(l,1) + 1),...
 round(bboxes(l,4) - bboxes(l,2) + 1));
     hold off;
     pause(1);
  end 
  nB = size(bboxes,1);
  rois = [rois [b*ones(1,nB) ; bboxes' ] ];
  maxH = max(maxH, size(imre{b},1));
  maxW = max(maxW, size(imre{b},2));
end
%fprintf('[maxH, maxW]: [%d, %d]', round(maxH), round(maxW));
imo = zeros(maxH,maxW,3,numel(batch),'single');
for b=1:numel(batch)
    imt = imre{b};
    if size(imt,3) == 1
        imt = cat(3, imt, imt, imt) ;
    end
    imre{b} = imt;
  % subtract mean
%  if ~isempty(opts.averageImage)
%    imre{b} = single(bsxfun(@minus,imre{b},opts.averageImage));
%  end
  sz = size(imre{b});
  imo(1:sz(1),1:sz(2),:,b) = single(imre{b});
end

