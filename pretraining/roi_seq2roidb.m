function  imdb = roi_seq2roidb(config, opts)
% SEQ2ROIDB
% Extract training bounding boxes from the sequence given by config, 
% to construct a dataset of RoIs for training MDNet.
%
% Hyeonseob Nam, 2015
% 
images = config.imgList;
gts = config.gt;

im = imread(images{1});
[h,w,~] = size(im);
imgSize = [h, w];

imdb = sample_rois(images, gts, imgSize, opts);



%--------------------------------------------------------------------------
function imdb = sample_rois(imgs, gts, imgSize, opts)
%--------------------------------------------------------------------------
fglabel = 2;
images.name = cell(length(gts), 1);
images.size = zeros(length(gts), 2); 
images.set = ones(length(gts),1);
boxes.gtbox = cell(length(gts), 1);
boxes.gtlabel = cell(length(gts),1);
boxes.plabel = cell(length(gts),1);
boxes.piou = cell(length(gts),1);
boxes.pbox = cell(length(gts),1);
%boxes.pgtidx = cell(length(gts),1);

for i=1:length(imgs)
    targetLoc = gts(i,:);
%     fprintf('sampling %s ...\n', images{idx});
    images.name{i} = imgs{i};
    images.size(i,:) = imgSize;
    boxes.gtlabel{i} =  fglabel;
    pos_examples = [targetLoc];
    if i >= opts.val_ratio* length(imgs)
	images.set(i) = 2;
    end
    piou = [1];
    while(size(pos_examples,1)<opts.posPerFrame-1)
        pos = genSamples(targetLoc, opts.posPerFrame*5,...
            imgSize, opts.scale_factor, 0.1, 5, false);
        r = overlap_ratio(pos,targetLoc);
        pos = pos(r>opts.posRange(1) & r<=opts.posRange(2),:);
        r = r(r > opts.posRange(1) & r <= opts.posRange(2));        
        if isempty(pos), continue; end
        l = length(pos);
        ids = randsample(l,min(l,opts.posPerFrame-size(pos_examples,1)));
        pos = pos(ids,:);
        pos_examples = [pos_examples;pos];
        piou = [piou; r(ids)];
    end
    neg_examples = [];
    niou = [];
    while(size(neg_examples,1)<opts.negPerFrame)
        neg = genSamples(targetLoc, opts.negPerFrame*2,...
            imgSize, opts.scale_factor, 0.8, 5, true);
        r = overlap_ratio(neg,targetLoc);
        neg = neg(r>=opts.negRange(1) & r<opts.negRange(2),:);
        r = r(r >= opts.negRange(1) & r < opts.negRange(2));        
        if isempty(neg), continue; end
        l = length(neg);
	ids = randsample(l,min(l,opts.negPerFrame-size(neg_examples,1)));
        neg = neg(ids,:);
        neg_examples = [neg_examples;neg];
        niou = [niou; r(ids)];
    end
    % convert to [x1,y1,x2,y2]
    pos_examples = [pos_examples(:,1:2), pos_examples(:,3) + pos_examples(:,1) - 1,...
			pos_examples(:,4) + pos_examples(:,2) - 1];
    neg_examples = [neg_examples(:,1:2), neg_examples(:,3) + neg_examples(:,1) - 1, ...
			neg_examples(:,4) + neg_examples(:,2) - 1];     
    boxes.gtbox{i} =  [ gts(i,1:2), gts(i,1:2) + gts(i,3:4) - 1];
    boxes.piou{i} = [piou; niou];
    boxes.plabel{i} = [2*ones(length(piou),1); ones(length(niou),1)];
    boxes.pbox{i} = vertcat(pos_examples, neg_examples);
%    boxes.pgtidx{i} = [(i+opts.offset)*ones(length(boxes.plabel{i}),1) ];
end
imdb.boxes = boxes;
imdb.images = images;


%--------------------------------------------------------------------------
function samples = genSamples(bb, n, imgSize, scale_factor, trans_range, scale_range, valid)
%--------------------------------------------------------------------------
h = imgSize(1); w = imgSize(2);

% [center_x center_y width height]
sample = [bb(1)+bb(3)/2 bb(2)+bb(4)/2, bb(3:4)];
samples = repmat(sample, [n, 1]);

samples(:,1:2) = samples(:,1:2) + trans_range*[bb(3)*(rand(n,1)*2-1) bb(4)*(rand(n,1)*2-1)];
samples(:,3:4) = samples(:,3:4) .* repmat(scale_factor.^(scale_range*(rand(n,1)*2-1)),1,2);
samples(:,3) = max(5,min(w-5,samples(:,3)));
samples(:,4) = max(5,min(h-5,samples(:,4)));

% [left top width height]
samples = [samples(:,1)-samples(:,3)/2 samples(:,2)-samples(:,4)/2 samples(:,3:4)];
if(valid)
    samples(:,1) = max(1,min(w-samples(:,3), samples(:,1)));
    samples(:,2) = max(1,min(h-samples(:,4), samples(:,2)));
    %samples(:,3) = min(w - samples(:,1), samples(:,3)) ;
    %samples(:,4) = min(h - samples(:,2), samples(:,4)) ;
else
    samples(:,1) = max(1-samples(:,3)/2,min(w-samples(:,3)/2, samples(:,1)));
    samples(:,2) = max(1-samples(:,4)/2,min(h-samples(:,4)/2, samples(:,2)));
    %samples(:,3) = min(w - samples(:,1), samples(:,3)) ;
    %samples(:,4) = min(h - samples(:,2), samples(:,4)) ;
end
samples = round(samples);

