function [ result, ts ] = mdnet_run(images, region, net, display)
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
acc = 0;
nFrames = length(images);

img = imread(images{1});
if(size(img,3)==1), img = cat(3,img,img,img); end
targetLoc = region;
result = zeros(nFrames, 4); result(1,:) = targetLoc;

[net_conv, net_fc, opts] = mdnet_init(img, net);
ts = [ts, {toc(initts) - acc}];
acc = acc + ts{end};
%% Train a bbox regressor
if(opts.bbreg)
    pos_examples = gen_samples('uniform_aspect', targetLoc, opts.bbreg_nSamples*10, opts, 0.3, 10);
    r = overlap_ratio(pos_examples,targetLoc);
    pos_examples = pos_examples(r>0.6,:);
    pos_examples = pos_examples(randsample(end,min(opts.bbreg_nSamples,end)),:);
    %plot_image(3, img, 0.1, [pos_examples(:,1:2), pos_examples(:,3:4)+pos_examples(:,1:2)-1]);
    feat_conv = mdnet_features_convX(net_conv, img, pos_examples, opts);
    
    %plot_feature_conv(2, feat_conv(:,:,1,:), 0.1, sprintf('Pos feature map of frame %d', 1));
    X = permute(gather(feat_conv),[4,3,1,2]);
    X = X(:,:);
    bbox = pos_examples;
    bbox_gt = repmat(targetLoc,size(pos_examples,1),1);
    bbox_reg = train_bbox_regressor(X, bbox, bbox_gt);
    %bbox_rec = predict_bbox_regressor(bbox_reg.model, X, bbox);

end
ts = [ts, {toc(initts) - acc}];
acc = acc + ts{end};
%% Extract training examples
fprintf('  extract features...\n');

% draw positive/negative samples
pos_examples = gen_samples('gaussian', targetLoc, opts.nPos_init*2, opts, 0.1, 5);
r = overlap_ratio(pos_examples,targetLoc);
pos_examples = pos_examples(r>opts.posThr_init,:);
pos_examples = pos_examples(randsample(end,min(opts.nPos_init,end)),:);

%neg_examples = [gen_samples('uniform', targetLoc, opts.nNeg_init, opts, 1, 10);...
%    gen_samples('whole', targetLoc, opts.nNeg_init, opts)];
neg_examples = [gen_samples('uniform', targetLoc, opts.nNeg_init*2, opts, 1, 10)];
    r = overlap_ratio(neg_examples,targetLoc);
neg_examples = neg_examples(r<opts.negThr_init,:);
neg_examples = neg_examples(randsample(end,min(opts.nNeg_init,end)),:);

examples = [pos_examples; neg_examples];
pos_idx = 1:size(pos_examples,1);
neg_idx = (1:size(neg_examples,1)) + size(pos_examples,1);
ts = [ts, {toc(initts) - acc}];
acc = acc + ts{end};
% extract conv3 features
feat_conv = mdnet_features_convX(net_conv, img, examples, opts);

pos_data = feat_conv(:,:,:,pos_idx);
neg_data = feat_conv(:,:,:,neg_idx);
ts = [ts, {toc(initts) - acc}];
acc = acc + ts{end};
target_conv = mdnet_features_convX(net_conv, img, targetLoc, opts);

%% Learning CNN
fprintf('  training cnn...\n');
[net_fc, poss,neg]= mdnet_finetune_hnm(net_fc, pos_data,neg_data,opts,...
    'maxiter',opts.maxiter_init,'learningRate',opts.learningRate_init);
 

ts = [ts, {toc(initts) - acc}];
acc = acc + ts{end};
%% Initialize displayots
if display
    figure(2);
    set(gcf,'Position',[200 100 600 400],'MenuBar','none','ToolBar','none');
    
    hd = imshow(img,'initialmagnification','fit'); hold on;
    rectangle('Position', targetLoc, 'EdgeColor', [1 0 0], 'Linewidth', 3);
    set(gca,'position',[0 0 1 1]);
    
    text(10,10,'1','Color','y', 'HorizontalAlignment', 'left', 'FontWeight','bold', 'FontSize', 30);
    hold off;
    drawnow;
    fname = sprintf('mdnet_%d.png', 1);
    saveas(gcf, fname);
end
%% Prepare training data for online update
total_pos_data = cell(1,1,1,nFrames);
total_neg_data = cell(1,1,1,nFrames);
% total_pos_data_export = cell(1,1,1,nFrames);
% total_neg_data_export = cell(1,1,1,nFrames);
neg_examples = gen_samples('uniform', targetLoc, opts.nNeg_update*2, opts, 2, 5);
r = overlap_ratio(neg_examples,targetLoc);
neg_examples = neg_examples(r<opts.negThr_init,:);
neg_examples = neg_examples(randsample(end,min(opts.nNeg_update,end)),:);

examples = [pos_examples; neg_examples];
pos_idx = 1:size(pos_examples,1);
neg_idx = (1:size(neg_examples,1)) + size(pos_examples,1);

feat_conv = mdnet_features_convX(net_conv, img, examples, opts);
%plot_feature_conv(3, cat(4, feat_conv(:,:,1,1:50), feat_conv(:,:,1,501:end)), 0.1, sprintf('Whole feature map of frame %d', 1));
total_pos_data{1} = feat_conv(:,:,:,pos_idx);
total_neg_data{1} = feat_conv(:,:,:,neg_idx);
%total_pos_data_export{1} = feat_conv(:,:,:,pos_idx);
%total_neg_data_export{1} = feat_conv(:,:,:,neg_idx);
success_frames = 1;
trans_f = opts.trans_f;
scale_f = opts.scale_f;
disp(toc(initts) - ts{end});
ts = [ts, {toc(initts) - acc}];
acc = acc + ts{end};

%% Main loop
bk = 9;
ts_ou_acc = zeros(bk,1);
ts_counts = zeros(bk,1);
for To = 2:nFrames;
    fprintf('Processing frame %d/%d... ', To, nFrames);
    ts_ou = zeros(bk,1);
    img = imread(images{To});
    if(size(img,3)==1), img = cat(3,img,img,img); end
    
    spf = tic;
    %% Estimation
    % draw target candidates
    samples = gen_samples('gaussian', targetLoc, opts.nSamples, opts, trans_f, scale_f);
    ts_ou(1) = toc(spf);
    feat_conv = mdnet_features_convX(net_conv, img, samples, opts);
    ts_ou(2) = toc(spf);
    % evaluate the candidates
    feat_fc = mdnet_features_fcX(net_fc, feat_conv, opts);
    feat_fc = squeeze(feat_fc)';
    [scores,idx] = sort(feat_fc(:,2),'descend');
    target_score = mean(scores(1:5));
    targetLoc = round(mean(samples(idx(1:5),:)));

    % final target
    result(To,:) = targetLoc;
    ts_ou(3) = toc(spf);

    % extend search space in case of failure
    if(target_score<0)
        trans_f = min(1.5, 1.1*trans_f);
    else
        trans_f = opts.trans_f;
    end
    fprintf('score: %.3f', target_score);
    % bbox regression
    if(opts.bbreg && target_score>0)
        X_ = permute(gather(feat_conv(:,:,:,idx(1:5))),[4,3,1,2]);
        %plot_feat = feat_conv(:,:,1:20,idx(1:5));
        %plot_feat = plot_feat(:,:,:);
        %plot_feature_conv(3, plot_feat, 0.1, sprintf('Pos feature map of frame %d', 1));
        X_ = X_(:,:);
        bbox_ = samples(idx(1:5),:);
        pred_boxes = predict_bbox_regressor(bbox_reg.model, X_, bbox_);
        result(To,:) = round(mean(pred_boxes,1));
    end
    ts_ou(4) = toc(spf);
    %% Prepare training data
    if(target_score>0)    
        pos_examples = gen_samples('gaussian', targetLoc, opts.nPos_update*2, opts, 0.1, 5);
        r = overlap_ratio(pos_examples,targetLoc);
        pos_examples = pos_examples(r>opts.posThr_update,:);
        pos_examples = pos_examples(randsample(end,min(opts.nPos_update,end)),:);
        
        neg_examples = gen_samples('uniform', targetLoc, opts.nNeg_update*2, opts, 2, 5);
        r = overlap_ratio(neg_examples,targetLoc);
        neg_examples = neg_examples(r<opts.negThr_update,:);
        neg_examples = neg_examples(randsample(end,min(opts.nNeg_update,end)),:);
        
        examples = [pos_examples; neg_examples];
        pos_idx = 1:size(pos_examples,1);
        neg_idx = (1:size(neg_examples,1)) + size(pos_examples,1);
        ts_ou(5) = toc(spf);
        feat_conv = mdnet_features_convX(net_conv, img, examples, opts);
        total_pos_data{To} = feat_conv(:,:,:,pos_idx);
        total_neg_data{To} = feat_conv(:,:,:,neg_idx);
        
%	total_pos_data_export{To} = total_pos_data{To};
%	total_neg_data_export{To} = total_neg_data{To};
        success_frames = [success_frames, To];
        if(numel(success_frames)>opts.nFrames_long)
            total_pos_data{success_frames(end-opts.nFrames_long)} = single([]);
        end
        if(numel(success_frames)>opts.nFrames_short)
            total_neg_data{success_frames(end-opts.nFrames_short)} = single([]);
        end
    	ts_ou(6) = toc(spf);
    else
        total_pos_data{To} = single([]);
        total_neg_data{To} = single([]);
%	total_pos_data_export{To} = total_pos_data{To};
%	total_neg_data_export{To} = total_neg_data{To};
    end

    %% Network update
    if((mod(To,opts.update_interval)==0 || target_score<0) && To~=nFrames)
        if (target_score<0) % short-term update
            pos_data = cell2mat(total_pos_data(success_frames(max(1,end-opts.nFrames_short+1):end)));
        else % long-term update
            pos_data = cell2mat(total_pos_data(success_frames(max(1,end-opts.nFrames_long+1):end)));
        end
        neg_data = cell2mat(total_neg_data(success_frames(max(1,end-opts.nFrames_short+1):end)));
        
%         fprintf('\n');
        [net_fc] = mdnet_finetune_hnm(net_fc,pos_data,neg_data,opts,...
            'maxiter',opts.maxiter_update,'learningRate',opts.learningRate_update);
    	ts_ou(7) = toc(spf);
    end
    
    spf = toc(spf);
    ts_ou(8) = spf;
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
        hc = get(gca, 'Children'); delete(hc(1:end-1));
        set(hd,'cdata',img); hold on;
        
        rectangle('Position', result(To,:), 'EdgeColor', [1 0 0], 'Linewidth', 3);
        set(gca,'position',[0 0 1 1]);
        
        text(10,10,num2str(To),'Color','y', 'HorizontalAlignment', 'left', 'FontWeight','bold', 'FontSize', 30); 
        hold off;
        drawnow;
        for l = 0:0.1:1
        if  To == round(100*l)
            fname = sprintf('mdnet_%d.png', To);
            saveas(gcf, fname);
        end
        end
    end
end

%% compute eclapsed time
ts_ou_acc = ts_ou_acc ./ ts_counts;
for l = 1:numel(ts_ou_acc)
ts = [ts,{ts_ou_acc(l)}];
end
