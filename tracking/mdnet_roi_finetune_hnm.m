function [net] = mdnet_roi_finetune_hnm(net, img_ori, img, targetLoc, pos_data, neg_data, varargin)
% MDNET_FINETUNE_HNM   
% Train a CNN by SGD, with hard minibatch mining.
%
% modified from cnn_train() in the MatConvNet library.
% Hyeonseob Nam, 2015
%

opts.useGpu = true;
opts.gpus = 3;
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

opts.batchSize = 128;
opts.batch_pos = 32;
opts.batch_neg = 96;
opts.derOutputs = {'losscls', 1} ;
opts.visualize = false;
[opts, args] = vl_argparse(opts, varargin) ;
% -------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------
ppred = find(arrayfun(@(a) strcmp(a.name, 'predclsf'), net.params)==1);
net.params(ppred).value = 0.01 * randn(1,1,size(net.params(ppred).value, 3), size(net.params(ppred).value, 4),'single');
net.params(ppred + 1).value = zeros(size(net.params(ppred + 1).value ), 'single');



%% initilizing
res = [] ;

n_pos = size(pos_data,1);
n_neg = size(neg_data,1);
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

if opts.useGpu
numGpus = numel(opts.gpus) ;
if numGpus >= 1
  net.move('gpu') ;
  state.momentum = cellfun(@gpuArray, state.momentum, 'uniformoutput', false) ;
end
if numGpus > 1
  parserv = ParameterServer(params.parameterServer) ;
  net.setParameterServer(parserv) ;
else
  parserv = [] ;
end
end

% objective fuction
objective = zeros(1,opts.maxiter);

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
        batch = neg_data(train_neg(hneg_start+(h-1)*opts.batchSize_hnm+1:hneg_start+h*opts.batchSize_hnm), :);
        rois = [ones(size(batch,1),1),  batch]' ;        img = single(img);
        rois = single(rois);
        % Evaluate network either on CPU or GPU.
        if opts.useGpu > 0
            img = gpuArray(img) ;
            rois = gpuArray(rois) ;
            net.move('gpu') ;
        end
        net.mode = 'test' ;
        net.conserveMemory = false ;
        inputs = {'x10', img, 'rois', rois} ;
        % backprop
        net.eval(inputs);
        % Extract class probabilities and  bounding box refinements
        probs = squeeze(gather(net.vars(net.getVarIndex('probcls')).value)) ;
        probs = probs(2,:);

        score_hneg((h-1)*opts.batchSize_hnm+1:h*opts.batchSize_hnm) = probs;
    end
    [~,ord] = sort(score_hneg,'descend');
    fprintf('max score: %.3f, min score: %.3f', score_hneg(ord(1)), score_hneg(ord(end)) );
    hnegs = train_neg(hneg_start+ord(1:opts.batch_neg));
    im_hneg = neg_data(hnegs, :);
    if ~opts.visualize
         figure(2);
         imshow(uint8(img_ori));
         hold on;
         bbox = [targetLoc;];
         for i=1:size(bbox,1)
            rectangle('Position',round([bbox(i,1),bbox(i,2), bbox(i,3), bbox(i,4)]), 'EdgeColor', 'r');
            hold on;
         end
         %bbox = neg_examples(1:100,:);
         bbox = im_hneg;
         for i=1:size(bbox,1)
            rectangle('Position',round([bbox(i,1),bbox(i,2), bbox(i,3), bbox(i,4)]), 'EdgeColor', 'b');
            hold on;
         end
         hold off;
         pause(0.1);
    end
    
 
%     fprintf('hnm: %d/%d, ', opts.batch_neg, opts.batchSize_hnm*opts.batchAcc_hnm) ;
   % ----------------------------------------------------------------------
    % get next image batch and labels
    % ----------------------------------------------------------------------
    pos = pos_data(train_pos((t-1)*opts.batch_pos+1:t*opts.batch_pos), :);
    batch = cat(1,pos,im_hneg);

    labels = [2*ones(opts.batch_pos,1,'single');ones(opts.batch_neg,1,'single')];
    btarget = bbox_transform(pos, repmat(targetLoc,size(pos,1),1));
    if opts.piecewise
    % regression error only for positives
       instance_weights = zeros(1,1,8,size(batch,1),'single');
       targets = zeros(1,1,8,size(batch,1),'single');
       targets(1,1,5:8,1:size(pos,1)) = btarget';
       instance_weights(1,1,5:8,1:size(pos,1)) = 1;
       if numel(opts.gpus) > 0
          targets = gpuArray(targets) ;
          instance_weights = gpuArray(instance_weights) ;
       end
    end


    rois = single([ones(size(batch, 1), 1), batch]');
    if opts.useGpu
        img = gpuArray(img);
        rois = gpuArray(rois) ;
        labels = gpuArray(labels);
    end
    % backprop
    opts.learningRate = opts.learningRate(min(t, numel(opts.learningRate)));
    params = opts;
    if opts.piecewise
        inputs = {'x10', img, 'rois', rois, 'label', labels, 'targets', targets, 'instance_weights', instance_weights } ;
    else
        inputs = {'x10', img, 'rois', rois, 'label', labels } ;
    end
    net.mode = 'normal' ;

    net.eval(inputs, params.derOutputs);
    % gradient step
    state = accumulateGradients(net, state, params, size(batch, 1), parserv) ;
   
    % print information
    loss = squeeze(gather(net.vars(net.getVarIndex('losscls')).value)) ;
    probs = squeeze(gather(net.vars(net.getVarIndex('probcls')).value)) ;

    objective(t) = gather(loss)/opts.batchSize ;
    iter_time = toc(iter_time);
    fprintf('iter: %d, objective %.3f, %.2f s\n', t, objective(t), iter_time) ;
    
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
