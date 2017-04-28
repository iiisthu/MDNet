function [ result, ts ] = mdnet_roi_run(images, region, net, display)
% MDNET_RUN
% Main interface for MDNet tracker
%
% INPUT:
%   images  - 1xN cell of the paths to image sequences
%   region  - 1x4 vector of the initial bounding box [left,top,width,height]
%   net     - The path to a trained MDNet
%   display - True for displying the tracking result
%
% OUTPUT:
%   result - Nx4 matrix of the tracking result Nx[left,top,width,height]
%
% Hyeonseob Nam, 2015
% 

if(nargin<4), display = true; end


%% Initialization
fprintf('Initialization...\n');
initts = tic;
ts = [];
nFrames = length(images);
targetLoc = [region(1:2), region(3:4) + region(1:2) - 1];
img = imread(images{1});
if(size(img,3)==1), img = cat(3,img,img,img); end
result = zeros(nFrames, 4); result(1,:) = region;
[net_conv, net_fc, opts] = mdnet_roi_init(img, net);
ts = [ts, {toc(initts)}];
if(opts.bbreg)
    pos_examples = roi_gen_samples('uniform_aspect', targetLoc, opts.bbreg_nSamples*10, opts, 0.3, 10);
    r = roi_overlap_ratio(pos_examples,targetLoc);
    pos_examples = pos_examples(r>0.6,:);
    pos_examples = pos_examples(randsample(end,min(opts.bbreg_nSamples,end)),:);
    [img_crop, target_crop, bboxes, R] = im_roi_crop(img, targetLoc, pos_examples, opts.crop_mode, opts.crop_size, opts.crop_padding, opts.minIn, opts.maxIn, []);
    if numel(opts.gpus) > 0
       img_crop = gpuArray(img_crop);
    end
    net_conv.mode = 'test';
    net_conv.eval({'input', img_crop});
    feat = squeeze(gather(net_conv.vars(net_conv.getVarIndex('x10')).value)) ; 
    rois = [ones(size(bboxes,1),1),  bboxes]' ;        
    rois = single(rois);
    % Evaluate network either on CPU or GPU.
    if numel(opts.gpus) > 0
       feat = gpuArray(feat);
       rois = gpuArray(rois) ;
    end
    net_fc.mode = 'test' ;
    net_fc.conserveMemory = false ;
    inputs = {'x10', feat, 'rois', rois} ;
    % backprop
    net_fc.eval(inputs);
    feat = squeeze(gather(net_fc.vars(net_fc.getVarIndex('xRP')).value)) ; 
    X = permute(gather(feat),[4,3,1,2]);
    X = X(:,:);
    bbox = pos_examples;
    bbox_gt = repmat(targetLoc,size(pos_examples,1),1);
    bbox_reg = train_bbox_regressor(X, bbox, bbox_gt);
end

%% Extract training examples
fprintf('  extract features...\n');

% draw positive/negative samples
pos_examples = roi_gen_samples('gaussian', targetLoc, opts.nPos_init*2, opts, 0.1, 5);
r = roi_overlap_ratio(pos_examples,targetLoc);
pos_examples = pos_examples(r>opts.posThr_init,:);
pos_examples = pos_examples(randsample(end,min(opts.nPos_init,end)),:);

neg_examples = [roi_gen_samples('uniform', targetLoc, opts.nNeg_init, opts, 1, 10);...
    roi_gen_samples('whole', targetLoc, opts.nNeg_init, opts)];
r = roi_overlap_ratio(neg_examples,targetLoc);
neg_examples = neg_examples(r<opts.negThr_init,:);
neg_examples = neg_examples(randsample(end,min(opts.nNeg_init,end)),:);


%% Learning CNN
fprintf('  training cnn...\n');
    if numel(opts.gpus) > 0
       window = gpuArray(single(img));
    end
 
net_conv.mode = 'test';
net_conv.eval({'input', window});
feat = squeeze(gather(net_conv.vars(net_conv.getVarIndex('x10')).value)) ; 

net_fc = mdnet_roi_finetune_hnm(net_fc,net_conv, {feat}, {targetLoc}, {pos_examples},{neg_examples},...
    'maxiter',opts.maxiter_init,'learningRate',opts.learningRate_init, ...
    'piecewise', opts.piecewise, 'derOutputs', opts.derOutputs, 'crop_mode', opts.crop_mode, ...
'crop_size', opts.crop_size, 'crop_padding',opts.crop_padding, 'maxIn', opts.maxIn, 'minIn', opts.minIn, 'gpus', opts.gpus);
ts = [ts, {toc(initts) - ts{end}}];
%% Initialize displayots
if display
    figure(1);
    set(gcf,'Position',[200 100 600 400],'MenuBar','none','ToolBar','none');
    
    hd = imshow(img,'initialmagnification','fit'); hold on;
    rectangle('Position', result(1,:), 'EdgeColor', [1 0 0], 'Linewidth', 3);
    set(gca,'position',[0 0 1 1]);
    
    text(10,10,'1','Color','y', 'HorizontalAlignment', 'left', 'FontWeight','bold', 'FontSize', 30);
    hold off;
    drawnow;
    fname = sprintf('fast_mdnet_%d.png', 1);
    saveas(gcf, fname);
end

%% Prepare training data for online update
total_pos_data = cell(1,nFrames);
total_neg_data = cell(1,nFrames);
total_img_data = cell(1,nFrames);
total_roi_data = cell(1,nFrames);

% total_pos_data_export = cell(1,1,1,nFrames);
% total_neg_data_export = cell(1,1,1,nFrames);
neg_examples = roi_gen_samples('uniform', targetLoc, opts.nNeg_update*2, opts, 2, 5);
r = roi_overlap_ratio(neg_examples,targetLoc);
neg_examples = neg_examples(r<opts.negThr_init,:);
neg_examples = neg_examples(randsample(end,min(opts.nNeg_update,end)),:);

total_img_data{1} = single(feat);
total_roi_data{1} = single(targetLoc);
total_pos_data{1} = single(pos_examples);
total_neg_data{1} = single(neg_examples);
%total_pos_data_export{1} = feat_conv(:,:,:,pos_idx);
%total_neg_data_export{1} = feat_conv(:,:,:,neg_idx);
success_frames = 1;
trans_f = opts.trans_f;
scale_f = opts.scale_f;
ts = [ts, {toc(initts) - ts{end}}];

%% Main loop
bk = 9;
net_fc.mode = 'test' ;

ts_ou_acc = zeros(bk,1);
ts_counts = zeros(bk,1);
target_score = 0;
for To = 2:nFrames;
    fprintf('Processing frame %d/%d... ', To, nFrames);
    ts_ou = zeros(bk,1);
    img = single(imread(images{To}));
    if(size(img,3)==1), img = cat(3,img,img,img); end
    
    spf = tic;
    %% Estimation
    % draw target candidnegs
    samples = roi_gen_samples('gaussian', targetLoc, opts.nSamples, opts, trans_f, scale_f);

    % evaluate the candidates
    [window,target_crop, bboxes,R] = im_roi_crop(img, targetLoc, samples, opts.crop_mode, opts.crop_size, opts.crop_padding, opts.minIn, opts.maxIn, []);
    bboxes = single([ones(size(bboxes, 1), 1, 'single'), bboxes]');

    bboxes_ori = bboxes;
    if numel(opts.gpus) > 0
        window = gpuArray(window) ;
        bboxes = gpuArray(bboxes) ;
    end
    net_conv.eval({'input', window});
    
    feat = squeeze(gather(net_conv.vars(net_conv.getVarIndex('x10')).value)) ;
    if numel(opts.gpus) > 0
        feat = gpuArray(feat);
    end

    inputs = {'x10', feat, 'rois', bboxes} ;
    % backprop
    net_fc.eval(inputs);
    probs = squeeze(gather(net_fc.vars(net_fc.getVarIndex('x17')).value)) ;
    cprobs = probs(2,:);
    if opts.piecewise
        pbbox = squeeze(gather(net_fc.vars(net_fc.getVarIndex('predbbox')).value)) ;
        cdeltas = pbbox(5:8,:);
        cdeltas = bsxfun(@plus,cdeltas',net_fc.meta.bboxMeanStd{1});
        cdeltas = bsxfun(@times,cdeltas,net_fc.meta.bboxMeanStd{2});
        cboxes = bbox_transform_inv(bboxes_ori(2:end,:)', cdeltas);
        
        %cls_dets = [cboxes cprobs'] ;
        %keep = bbox_nms(cls_dets, opts.nmsThreshold) ;
        [~, keep] = sort(cprobs,'descend');
        cls_dets = [cboxes(keep, :) cprobs(keep)'];
        sel_boxes = find(cls_dets(:,end) >= opts.confThreshold) ;
        if ~isempty(sel_boxes)
        % final target 
            targetLoc = [cls_dets(sel_boxes(1:min(5,end)),1:4);];
            target_score = mean(cls_dets(sel_boxes(1:min(5,end)),end));
            targetLoc = [targetLoc./repmat([R(3:4), R(3:4)],size(targetLoc,1),1) + repmat([R(1:2),R(1:2)], size(targetLoc, 1),1) ];
        else
            thisLoc = [cls_dets(1, 1:end-1)./[R(3:4), R(3:4)] + [R(1:2),R(1:2)]];
            %w = 1/(1+exp(target_score - cls_dets(1,end)));
            %targetLoc = 2*[(1-w)* targetLoc; w*thisLoc];
            targetLoc = [thisLoc];
            target_score = cls_dets(1,end);
        end
    else
        [sprobs ,ord]  = sort(cprobs, 'descend');
        sel_boxes = find(sprobs >= opts.confThreshold);
        if ~isempty(sel_boxes)
            targetLoc = bboxes_ori(2:end, ord(sel_boxes(1:min(3,end))))';
            target_score = mean(sprobs(sel_boxes(1:min(3,end))));
            % bbox regression
            if(opts.bbreg && target_score>opts.scoreThr)
               feat_rois = squeeze(gather(net_fc.vars(net_fc.getVarIndex('xRP')).value)) ;
               X_ = permute(gather(feat_rois(:,:,:,ord(sel_boxes(1:min(5,end))))),[4,3,1,2]);
               X_ = X_(:,:);
               bbox_ = bboxes_ori(2:end,ord(sel_boxes(1:min(5,end))))';
               targetLoc = predict_bbox_regressor(bbox_reg.model, X_, bbox_);
            end

            targetLoc = [targetLoc./repmat([R(3:4), R(3:4)],size(targetLoc,1),1) + repmat([R(1:2),R(1:2)], size(targetLoc, 1),1) ];
        else
            thisLoc = [bboxes_ori(2:end, ord(1))'./[R(3:4), R(3:4)] + [R(1:2),R(1:2)]];
            w = 1/(1+exp(target_score - sprobs(ord(1))));
            %targetLoc = 2*[(1-w)* targetLoc; w*thisLoc];
            targetLoc = [thisLoc];
            %target_score = (1-w)*target_score + w*sprobs(ord(1));
            target_score = sprobs(ord(1));
        end
    end
    fprintf('max probs: %.3f, target_score:%.3f\t', max(cprobs), target_score);

    if opts.visualize
     figure(2);
     imshow(uint8(window));
     hold on;
     bbox = [target_crop;];
     for i=1:size(bbox,1)
        rectangle('Position',round([bbox(i,1),bbox(i,2), bbox(i,3) - bbox(i,1) + 1, bbox(i,4) - bbox(i,2) + 1]), 'EdgeColor', 'r');
        hold on;
     end
      pause(0.1);
     bbox = targetLoc;
     for i=1:size(bbox,1)
        rectangle('Position',round([bbox(i,1),bbox(i,2), bbox(i,3) - bbox(i,1) + 1, bbox(i,4) - bbox(i,2) + 1]), 'EdgeColor', 'y');
        hold on;
     end
     pause(0.1);
     %bbox = neg_examples(1:100,:);
     bbox = bboxes_ori(2:end,:)';
     for i=1:size(bbox,1)
        rectangle('Position',round([bbox(i,1),bbox(i,2), bbox(i,3) - bbox(i,1) + 1, bbox(i,4) - bbox(i,2) + 1]), 'EdgeColor', 'b');
        hold on;
     end
     hold off;
     pause(0.1);
    end
    targetLoc = round(mean(targetLoc,1));
    targetLoc = max(1, targetLoc);
    result(To,:) = [targetLoc(1:2), targetLoc(3:4) - targetLoc(1:2) + 1];
    % extend search space in case of failure
    if(target_score <= opts.scoreThr)
        trans_f = min(1.5, 1.1*trans_f);
        %scale_f = min(1.5, 1.1*scale_f);
    else
        trans_f = opts.trans_f;
    end
    
    %% Prepare training data
    if(target_score> 0)    
        pos_examples = roi_gen_samples('gaussian', targetLoc, opts.nPos_update*2, opts, 0.1, 5);
        r = roi_overlap_ratio(pos_examples,targetLoc);
        pos_examples = pos_examples(r>opts.posThr_update,:);
        pos_examples = pos_examples(randsample(end,min(opts.nPos_update,end)),:);
        if size(pos_examples, 1) == 0
            continue;
        end
        neg_examples = roi_gen_samples('uniform', targetLoc, opts.nNeg_update*2, opts, 2, 5);
        r = roi_overlap_ratio(neg_examples, targetLoc);
        neg_examples = neg_examples(r<opts.negThr_update,:);
        neg_examples = neg_examples(randsample(end,min(opts.nNeg_update,end)),:);
        
        if numel(opts.gpus) > 0
           window = gpuArray(img) ;
        end
        net_conv.eval({'input', window});
        feat = squeeze(gather(net_conv.vars(net_conv.getVarIndex('x10')).value)) ;
       
        total_pos_data{To} = pos_examples;
        total_neg_data{To} = neg_examples;
        total_img_data{To} = feat; 
        total_roi_data{To} = single(targetLoc);

%       total_pos_data_export{To} = total_pos_data{To};
%       total_neg_data_export{To} = total_neg_data{To};
        success_frames = [success_frames, To];
        if(numel(success_frames)>opts.nFrames_long)
            total_pos_data{success_frames(end-opts.nFrames_long)} = single([]);
            total_img_data{success_frames(end-opts.nFrames_long)} = single([]);
            total_roi_data{success_frames(end-opts.nFrames_long)} = single([]);
            total_neg_data{success_frames(end-opts.nFrames_long)} = single([]);
        end
    else
        total_pos_data{To} = single([]);
        total_neg_data{To} = single([]);
        total_img_data{To} = single([]);
        total_roi_data{To} = single([]);
%       total_pos_data_export{To} = total_pos_data{To};
%       total_neg_data_export{To} = total_neg_data{To};
    end

    %% Network update
    if((mod(To,opts.update_interval)==0 || target_score<opts.scoreThr) && To~=nFrames)
        if (target_score<opts.scoreThr) % short-term update
            pos_data = total_pos_data(success_frames(max(1,end-opts.nFrames_short+1):end));
            img_data = total_img_data(success_frames(max(1,end-opts.nFrames_short+1):end));
            roi_data = total_roi_data(success_frames(max(1,end-opts.nFrames_short+1):end));
            neg_data = total_neg_data(success_frames(max(1,end-opts.nFrames_short+1):end));
        else % long-term update
            pos_data = total_pos_data(success_frames(max(1,end-opts.nFrames_long+1):end));
            img_data = total_img_data(success_frames(max(1,end-opts.nFrames_long+1):end));
            roi_data = total_roi_data(success_frames(max(1,end-opts.nFrames_long+1):end));
            neg_data = total_neg_data(success_frames(max(1,end-opts.nFrames_long+1):end));
        end
        
%         fprintf('\n');
        
        sus = success_frames(max(1,end-numel(pos_data)+1):end);
        % generate batch
        img_update = cell(floor(numel(img_data)/opts.batchimg) + 1, 1);
        pos_update = cell(floor(numel(img_data)/opts.batchimg) + 1, 1);
        neg_update = cell(floor(numel(img_data)/opts.batchimg) + 1, 1);
        roi_update = cell(floor(numel(img_data)/opts.batchimg) + 1, 1);
        for j = 1:numel(img_data)
            offset = floor( (j-1)/opts.batchimg ) + 1;
            img_update{offset}{mod(j-1, opts.batchimg) + 1} = img_data{j};
            pos_update{offset}{mod(j-1, opts.batchimg) + 1} = pos_data{j};
            neg_update{offset}{mod(j-1, opts.batchimg) + 1} = neg_data{j};
            roi_update{offset}{mod(j-1, opts.batchimg) + 1} = roi_data{j};
        end
        for j = 1:numel(img_update)
       	    net_fc = mdnet_roi_finetune_hnm(net_fc,net_conv, img_update{j},roi_update{j}, pos_update{j},neg_update{j},...
                                        'maxiter',opts.maxiter_update,'learningRate',opts.learningRate_update, ...
					'piecewise', opts.piecewise, 'derOutputs', opts.derOutputs, 'crop_mode', opts.crop_mode, ...
					'crop_size', opts.crop_size, 'crop_padding',opts.crop_padding, ...
					'maxIn', opts.maxIn, 'minIn', opts.minIn, 'gpus', opts.gpus);
        end
    end
    
    spf = toc(spf);
    fprintf('%f seconds\n',spf);
    %% compute eclapsed time
    acc = 0;
    for s = 2:bk-2
        acc = acc + ts_ou(s-1);
        if ts_ou(s) > 0
            ts_ou(s) = ts_ou(s) - acc;
        end
    end
    ts_ou(bk) = ts_ou(bk-1) - ts_ou(7);
    ts_counts = ts_counts + double(ts_ou > 0);
    ts_ou_acc = ts_ou_acc + ts_ou;
    %% Display
    if display
        figure(1);
        hc = get(gca, 'Children'); delete(hc(1:end-1));
        set(hd,'cdata',uint8(img)); hold on;
        disp(result(To,:));
        rectangle('Position',  result(To,:), 'EdgeColor', [1 0 0], 'Linewidth', 3);
        set(gca,'position',[0 0 1 1]);
        
        text(10,10,num2str(To),'Color','y', 'HorizontalAlignment', 'left', 'FontWeight','bold', 'FontSize', 30); 
        hold off;
        for l = 0:0.1:1
        if  To == round(100*l)
            fname = sprintf('fast_mdnet_%d.png', To);
            saveas(gcf, fname);
        end
        end
        drawnow;
    end
end

%% compute eclapsed time
ts_ou_acc = ts_ou_acc ./ ts_counts;
for l = 1:numel(ts_ou_acc)
ts = [ts,{ts_ou_acc(l)}];
end
