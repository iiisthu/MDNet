function [net] = mdnet_roi_finetune_hnm(net, img, img_ori, targetLoc, pos_data, neg_data, varargin)
% MDNET_FINETUNE_HNM   
% Train a CNN by SGD, with hard minibatch mining.
%
% modified from cnn_train() in the MatConvNet library.
% Hyeonseob Nam, 2015
%

opts.gpus = [];
opts.conserveMemory = true ;
opts.sync = true ;
opts.piecewise = 0;
opts.maxiter = 30;
opts.learningRate = 0.001;
opts.weightDecay = 0.0005 ;
opts.momentum = 0.9 ;
opts.nesterovUpdate = false ;
opts.batchSize_hnm = 256;
opts.batchAcc_hnm = 4;
opts.parameterServer.method = 'mmap' ;
opts.parameterServer.prefix = 'mcn' ;
opts.batchSize = 128;
opts.batch_pos = 32;
opts.batch_neg = 96;
opts.derOutputs = {'losscls', 1} ;
opts.debug = false;

[opts, args] = vl_argparse(opts, varargin) ;
% -------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------
%ppred = find(arrayfun(@(a) strcmp(a.name, 'predclsf'), net.params)==1);
%net.params(ppred).value = 0.01 * randn(1,1,size(net.params(ppred).value, 3), size(net.params(ppred).value, 4),'single');
%net.params(ppred + 1).value = zeros(size(net.params(ppred + 1).value ), 'single');
if numel(img) == 0
   return;
end


%% initilizing
res = [] ;
n_pos = 0;
n_neg = 0;
new_pos_data = [];
new_neg_data = [];
for i = 1:numel(pos_data)
if numel(pos_data) == 0
   continue;
end
n_pos = n_pos + size(pos_data{i},1);
n_neg = n_neg + size(neg_data{i},1);
new_pos_data = cat(1,new_pos_data, [i*ones(size(pos_data{i}, 1),1) pos_data{i}]);
new_neg_data = cat(1,new_neg_data, [i*ones(size(neg_data{i}, 1),1) neg_data{i}]);
end


train_pos_cnt = 0;
train_neg_cnt = 0;

% extract positive batches
train_pos = [];
remain = opts.batch_pos*opts.maxiter;
while(remain>0)
    if(train_pos_cnt==0)
        train_pos_list = randperm(n_pos)';
    end
    train_pos = cat(1,train_pos,...
        train_pos_list(train_pos_cnt+1:min(end,train_pos_cnt+remain)));
    train_pos_cnt = min(length(train_pos_list),train_pos_cnt+remain);
    train_pos_cnt = mod(train_pos_cnt,length(train_pos_list));
    remain = opts.batch_pos*opts.maxiter-length(train_pos);
end

% extract negative batches
train_neg = [];
remain = opts.batchSize_hnm*opts.batchAcc_hnm*opts.maxiter;
while(remain>0)
    if(train_neg_cnt==0)
        train_neg_list = randperm(n_neg)';
    end
    train_neg = cat(1,train_neg,...
        train_neg_list(train_neg_cnt+1:min(end,train_neg_cnt+remain)));
    train_neg_cnt = min(length(train_neg_list),train_neg_cnt+remain);
    train_neg_cnt = mod(train_neg_cnt,length(train_neg_list));
    remain = opts.batchSize_hnm*opts.batchAcc_hnm*opts.maxiter-length(train_neg);
end

% learning rate
lr = opts.learningRate ;

% for saving positives

% for saving hard negatives
state = [];
if isempty(state) || isempty(state.momentum)
  state.momentum = num2cell(zeros(1, numel(net.params))) ;
end

numGpus = numel(opts.gpus) ;
if numGpus >= 1
  net.move('gpu') ;
  state.momentum = cellfun(@gpuArray, state.momentum, 'uniformoutput', false) ;
end
if numGpus > 1
  parserv = ParameterServer(opts.parameterServer) ;
  net.setParameterServer(parserv) ;
else
  parserv = [] ;
end

% objective fuction
loss_cls = zeros(1,opts.maxiter);
loss_bbox = zeros(1,opts.maxiter);
img_batch = zeros(size(img{1},1), size(img{1}, 2), size(img{1}, 3), numel(img), 'single');
for i = 1:numel(img)
   img_batch(:,:,:,i) = single(img{i});
end
%% training on training set
% fprintf('\n');
for t=1:opts.maxiter
%     fprintf('\ttraining batch %3d of %3d ... ', t, opts.maxiter) ;
    iter_time = tic ;
    
    % ----------------------------------------------------------------------
    % hard negative mining
    % ----------------------------------------------------------------------
    score_hneg = zeros(opts.batchSize_hnm*opts.batchAcc_hnm,1);
    hneg_start = opts.batchSize_hnm*opts.batchAcc_hnm*(t-1);
    for h=1:opts.batchAcc_hnm
        hnm_s = tic;
        batch = new_neg_data(train_neg(hneg_start+(h-1)*opts.batchSize_hnm+1:hneg_start+h*opts.batchSize_hnm), :);
        neg_batch = [];
        target_batch = [];
        img_tmp = {};
        for i = 1:numel(img)
             matched_neg = find( batch(:,1) == i );
	     neg_batch = cat(1,neg_batch, batch(matched_neg, :));
             target_batch = cat(1,target_batch, targetLoc{i});
        end
        rois = single(neg_batch)';        
    %    fprintf('dirty_ops %.2f\n',toc(hnm_s));
        % Evaluate network either on CPU or GPU.
        if numel(opts.gpus) > 0
	    feat = gpuArray(img_batch);
            rois = gpuArray(rois) ;
        end
        net.mode = 'test' ;
        net.conserveMemory = false ;
        inputs = {'x10', feat, 'rois', rois} ;
        % backprop
        net.eval(inputs);
        % Extract class probabilities and  bounding box refinements
        probs = squeeze(gather(net.vars(net.getVarIndex('probcls')).value)) ;
        probs = probs(2,:);
        net.reset();
        score_hneg((h-1)*opts.batchSize_hnm+1:h*opts.batchSize_hnm) = probs;
    end
    [score_sorted,ord] = sort(score_hneg,'descend');
    %fprintf('max score: %.3f, min score: %.3f', score_hneg(ord(1)), score_hneg(ord(end)) );
    hnegs = train_neg(hneg_start+ord(1:opts.batch_neg));
    im_hneg = new_neg_data(hnegs, :);
    hntime = toc(iter_time);
%     fprintf('hnm: %d/%d, ', opts.batch_neg, opts.batchSize_hnm*opts.batchAcc_hnm) ;
   % ----------------------------------------------------------------------
    % get next image batch and labels
    % ----------------------------------------------------------------------
    pos = new_pos_data(train_pos((t-1)*opts.batch_pos+1:t*opts.batch_pos), :);
    batch = cat(1,pos,im_hneg);
    bbox_batch = [];
    target_batch = [];
    label_batch =[];
    for l = 1:numel(img)
         matched = find( batch(:,1) == l );
         label_batch = [label_batch; 2*ones(numel(find(matched <= opts.batch_pos)), 1, 'single'); ones(numel(find(matched > opts.batch_pos)), 1, 'single')];
         bbox_batch = cat(1,bbox_batch, batch(matched, :));
         target_batch = cat(1,target_batch, targetLoc{l});
         if opts.debug && t == opts.maxiter
             pos = batch(matched(matched <= opts.batch_pos), 2:end);
             neg = batch(matched(matched > opts.batch_pos), 2:end);
             %plot_image(7, img_ori{l}, 0.1, pos, neg);
         end
     end
    if opts.piecewise
       pos_in_batch = find(label_batch == 2);
       btargets = zeros(size(pos_in_batch,1), 4, 'single');
       for i = 1:numel(img)
          tid = find(bbox_batch(pos_in_batch, 1) == i);
          if numel(img) == 1
             btargets(tid, :) = repmat(single(target_batch), numel(tid), 1);
          else
             btargets(tid, :) = repmat(single(target_batch(i,:)), numel(tid),1);
          end
       end
       btarget = bbox_transform(bbox_batch(pos_in_batch,2:end),btargets);
    % regression error only for positives
       btarget = bsxfun(@minus,btarget,net.meta.bboxMeanStd{1});
       btarget = bsxfun(@rdivide,btarget,net.meta.bboxMeanStd{2});
       instance_weights = zeros(1,1,8,size(batch,1),'single');
       targets = zeros(1,1,8,size(batch,1),'single');
       targets(1,1,5:8,pos_in_batch) = btarget';
       instance_weights(1,1,5:8,pos_in_batch) = 1;
       if numel(opts.gpus) > 0
          targets = gpuArray(targets) ;
          instance_weights = gpuArray(instance_weights) ;
       end
    end
    bbox_batch = single(bbox_batch');
    feat = img_batch;
    if numel(opts.gpus) > 0
        feat = gpuArray(feat);
        bbox_batch = gpuArray(bbox_batch) ;
        label_batch = gpuArray(label_batch);
    end
    % backprop
    opts.learningRate = opts.learningRate(min(t, numel(opts.learningRate)));
    params = opts;
    if opts.piecewise
        inputs = {'x10', feat, 'rois', bbox_batch, 'label', label_batch, 'targets', targets, 'instance_weights', instance_weights } ;
    else
        inputs = {'x10', feat, 'rois', bbox_batch , 'label', label_batch } ;
    end
    net.mode = 'normal' ;

    net.eval(inputs, params.derOutputs);
    % gradient step
    if ~isempty(parserv), parserv.sync() ; end
    state = accumulateGradients(net, state, params, size(batch, 1), parserv) ;
   
    % print information
    loss = squeeze(gather(net.vars(net.getVarIndex('losscls')).value)) ;
    probs = squeeze(gather(net.vars(net.getVarIndex('probcls')).value)) ;
    if opts.debug && ( t == 1 || t == opts.maxiter)
        features = squeeze(gather(net.vars(net.getVarIndex('xRP')).value)) ;
        plot_feature_conv(8, features(:,:,2,:), 0.1, 'Training: bbox fmap');
    end
    loss_cls(t) = gather(loss)/opts.batchSize ;
    if opts.piecewise
    	lossb = squeeze(gather(net.vars(net.getVarIndex('lossbbox')).value)) ;
        probbox = squeeze(gather(net.vars(net.getVarIndex('predbbox')).value)) ;
        bbox_target = squeeze(gather(targets));
    	loss_bbox(t) = gather(lossb)/opts.batchSize ;
    end
    iter_time = toc(iter_time);
    fprintf('iter: %d, cls_loss %.3f, bbox_loss: %.6f, time %.2f, hntime %.2f s\n', t, loss_cls(t), loss_bbox(t), iter_time, hntime) ;
    net.reset();
    
end % next batch
net.reset();

% -------------------------------------------------------------------------
function state = accumulateGradients(net, state, params, batchSize, parserv)
% -------------------------------------------------------------------------
numGpus = numel(params.gpus) ;
otherGpus = setdiff(1:numGpus, labindex) ;
for p=1:numel(net.params)
  if ~ params.piecewise
      expel = 'predbbox';
      if strncmpi(net.params(p).name, expel, length(expel))
        continue;
      end
  end
  if ~isempty(parserv)
    parDer = parserv.pullWithIndex(p) ;
  else
    parDer = net.params(p).der ;
  end

  switch net.params(p).trainMethod

    case 'average' % mainly for batch normalization
      thisLR = net.params(p).learningRate ;
      net.params(p).value = vl_taccum(...
          1 - thisLR, net.params(p).value, ...
          (thisLR/batchSize/net.params(p).fanout),  parDer) ;

    case 'gradient'
      thisDecay = params.weightDecay * net.params(p).weightDecay ;
      thisLR = params.learningRate * net.params(p).learningRate ;

      if thisLR>0 || thisDecay>0
        % Normalize gradient and incorporate weight decay.
        parDer = vl_taccum(1/batchSize, parDer, ...
                           thisDecay, net.params(p).value) ;

        % Update momentum.
        state.momentum{p} = vl_taccum(...
          params.momentum, state.momentum{p}, ...
          -1, parDer) ;
        % Nesterov update (aka one step ahead).
        if params.nesterovUpdate
          delta = vl_taccum(...
            params.momentum, state.momentum{p}, ...
            -1, parDer) ;
        else
          delta = state.momentum{p} ;
        end

        % Update parameters.
        net.params(p).value = vl_taccum(...
          1,  net.params(p).value, thisLR, delta) ;
      end
    otherwise
      error('Unknown training method ''%s'' for parameter ''%s''.', ...
        net.params(p).trainMethod, ...
        net.params(p).name) ;
  end
end
