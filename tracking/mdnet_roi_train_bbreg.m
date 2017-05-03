function bbox_reg =  mdnet_roi_train_bbreg(net_conv, net_fc, img, targetLoc, opts)
    pos_examples = roi_gen_samples('uniform_aspect', targetLoc, opts.bbreg_nSamples*10, opts, 0.2, 5);
    r = roi_overlap_ratio(pos_examples,targetLoc);
    pos_examples = pos_examples(r>0.6,:);
    pos_examples = pos_examples(randsample(end,min(opts.bbreg_nSamples,end)),:);
    range = (2 + opts.scale_factor^ 5);
    [img_crop, target_crop, bboxes, R] = im_roi_crop(img, targetLoc, pos_examples, opts.crop_mode, opts.crop_size, opts.crop_padding, 1, range, []);
    %plot_image(3, img_crop, 0.1, bboxes);
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
    %plot_feature_conv(2, feat(:,:,2,:), 0.1, sprintf('Pos feature map of frame %d', 1));
    X = permute(gather(feat),[4,3,1,2]);
    X = X(:,:);
    % should use [x,y,w,h] representation
    bbox = [bboxes(:,1:2), bboxes(:,3:4) - bboxes(:,1:2) + 1];
    bbox_gt = repmat([target_crop(1:2), target_crop(3:4) - target_crop(1:2) + 1],size(pos_examples,1),1);
    bbox_reg = train_bbox_regressor(X, bbox, bbox_gt);
    bbox_rec = predict_bbox_regressor(bbox_reg.model, X, bbox);
end
