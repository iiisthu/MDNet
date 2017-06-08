function [ feat ] = mdnet_features_convX(net, img, boxes, opts)
% MDNET_FEATURES_CONVX
% Extract CNN features from bounding box regions of an input image.
%
% Hyeonseob Nam, 2015
% 

n = size(boxes,1);
ims = mdnet_extract_regions(img, boxes, opts);
nBatches = ceil(n/opts.batchSize_test);
up = 180;
for i=1:nBatches
%     fprintf('extract batch %d/%d...\n',i,nBatches);
    
    batch = ims(:,:,:,opts.batchSize_test*(i-1)+1:min(end,opts.batchSize_test*i));
    if opts.debug && i == 1
        test_image = batch(:,:,:,1:2:min(up:end));
        plot_feature_conv(3, test_image, 1, 'image_patches');
    end
    if(opts.useGpu)
        batch = gpuArray(batch);
    end
    
    res = vl_simplenn(net, batch, [], [], ...
        'disableDropout', true, ...
        'conserveMemory', false, ...
        'sync', true) ;
    
    f = gather(res(end).x) ;
    if ~exist('feat','var')
        feat = zeros(size(f,1),size(f,2),size(f,3),n,'single');
    end
    feat(:,:,:,opts.batchSize_test*(i-1)+1:min(end,opts.batchSize_test*i)) = f;
    if opts.debug && i == 1
        show_f = gather(res(6).x);
        [ ~, ~, ~, channel] = size(show_f); 
        sum_act = sum(reshape(show_f, [], channel));
        [~, id] = max(sum_act(1:2:min(up,end)));
        disp(id);
        plot_feature_conv(4, squeeze(show_f(:,:,1:min(90,end),1)), 1, 'feature comparison');
        plot_feature_conv(5, squeeze(show_f(:,:,1,1:2:min(up,end))), 1, 'single feature channel multiple samples');
    end
end
