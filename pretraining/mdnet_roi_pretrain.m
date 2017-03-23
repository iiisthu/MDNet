function [ ] = mdnet_roi_pretrain( varargin )
% MDNET_PRETRAIN
% Pretrain MDNet from multiple tracking sequences.
%
% Modified from cnn_imagenet() in the MatConvNet library.
% Hyeonseob Nam, 2015
% 

% The list of tracking sequences for training MDNet.
opts.seqsList  = {struct('dataset','vot2013','list','pretraining/seqList/vot13-otb.txt'),...
    struct('dataset','vot2014','list','pretraining/seqList/vot14-otb.txt'),...
    struct('dataset','vot2015','list','pretraining/seqList/vot15-otb.txt')};

% The path to the initial network. 
opts.netFile    = fullfile('models','mdnet_roi_init_piecewise.mat') ;
%opts.netFile = fullfile('data','bkp1','net-epoch-11.mat');
% The path to the output MDNet model.
opts.outFile     = fullfile('models','mdnet_roi_vot-otb_new.mat') ;
opts.expDir      = 'models';
% The directory to store the RoIs for training MDNet.
opts.imdbDir     = fullfile('models','data_vot-otb') ;

opts.sampling.crop_mode         = 'warp';
opts.sampling.numFetchThreads   = 8 ;
opts.sampling.posRange          = [0.7 1];
opts.sampling.negRange          = [0 0.5];
opts.sampling.input_size        = 107;
opts.sampling.crop_padding      = 16;

opts.sampling.posPerFrame       = 50;
opts.sampling.negPerFrame       = 200;
opts.sampling.scale_factor      = 1.05;
opts.sampling.flip              = false;
opts.sampling.val_ratio         = 0.9;
% fast rcnn parameters
opts.train.batchSize        = 2 ;

opts.train.numEpochs        = 100 ; % #cycles (#iterations/#domains)
opts.train.learningRate     = 0.01*[0.001*ones(1,1); 0.00001*ones(50,1);0.000001*ones(49,1)] ; % x10 for fc4-6
opts.train.gpus = [] ;
opts.train.numSubBatches = 1 ;
opts.train.prefetch = false ; % does not help for two images in a batch
opts.train.weightDecay = 0.0005 ;
opts.train.expDir = fullfile('data','exp_piecewise') ;
opts.piecewise = 1;

opts.numFetchThreads = 2 ;
opts.train.numEpochs = numel(opts.train.learningRate) ;
opts = vl_argparse(opts, varargin) ;
opts.imdbPath  = fullfile(opts.imdbDir, 'roi_imdb.mat');
genDir(opts.imdbDir) ;

%% Sampling training data
if exist(opts.imdbPath,'file')
    load(opts.imdbPath) ;
else
    imdb = mdnet_roi_setup_data(opts.seqsList, opts.sampling);
    save(opts.imdbPath, 'imdb') ;
end
if opts.piecewise 
opts.train.derOutputs = {'losscls', 1, 'lossbbox', 1} ;
else
opts.train.derOutputs = {'losscls', 1} ;
end
%% Initializing MDNet
net = mdnet_roi_init_train(opts);

%% Training MDNet
% minibatch options
bopts.useGpu = numel(opts.train.gpus) >  0 ;
bopts.maxScale = 1000;
bopts.scale = 600;
bopts.batch_pos        = 32;
bopts.batch_neg        = 96;
bopts.bgLabel = 1;
bopts.piecewise = opts.piecewise;
bopts.visualize = 0;
bopts.interpolation = net.meta.normalization.interpolation;
bopts.numThreads = opts.numFetchThreads;
bopts.prefetch = opts.train.prefetch;


[net,info] = cnn_train_dag(net, imdb, @(i,k,b) ...
                           getBatch(bopts,i,k,b), ...
                           opts.train) ;

% --------------------------------------------------------------------
%                                                               Deploy
% --------------------------------------------------------------------
modelPath = fullfile(opts.expDir, 'net-deployed.mat');
if ~exist(modelPath,'file')
  net = deployFRCNN(net,imdb);
  net_ = net.saveobj() ;
  save(modelPath, '-struct', 'net_') ;
  clear net_ ;
end

function net = deployFRCNN(net,imdb)
% --------------------------------------------------------------------
% function net = deployFRCNN(net)
for l = numel(net.layers):-1:1
  if isa(net.layers(l).block, 'dagnn.Loss') || ...
      isa(net.layers(l).block, 'dagnn.DropOut')
    layer = net.layers(l);
    net.removeLayer(layer.name);
    net.renameVar(layer.outputs{1}, layer.inputs{1}, 'quiet', true) ;
  end
end

net.rebuild();

pfc8 = net.getLayerIndex('predcls') ;
net.addLayer('probcls',dagnn.SoftMax(),net.layers(pfc8).outputs{1},...
  'probcls',{});

net.vars(net.getVarIndex('probcls')).precious = true ;

idxBox = net.getLayerIndex('predbbox') ;
if ~isnan(idxBox)
  net.vars(net.layers(idxBox).outputIndexes(1)).precious = true ;
  % incorporate mean and std to bbox regression parameters
  blayer = net.layers(idxBox) ;
  filters = net.params(net.getParamIndex(blayer.params{1})).value ;
  biases = net.params(net.getParamIndex(blayer.params{2})).value ;

  boxMeans = single(imdb.boxes.bboxMeanStd{1}');
  boxStds = single(imdb.boxes.bboxMeanStd{2}');

  net.params(net.getParamIndex(blayer.params{1})).value = ...
    bsxfun(@times,filters,...
    reshape([boxStds(:)' zeros(1,4,'single')]',...
    [1 1 1 8]));

  biases = biases .* [boxStds(:)' zeros(1,4,'single')];

  net.params(net.getParamIndex(blayer.params{2})).value = ...
    bsxfun(@plus,biases, [boxMeans(:)' zeros(1,4,'single')]);
end

net.mode = 'test' ;

% -------------------------------------------------------------------------
function inputs = getBatch(opts, imdb, k,  batch)
% -------------------------------------------------------------------------
if isempty(batch)
  return;
end
opts.batch_pos        = floor(opts.batch_pos/numel(batch));
opts.batch_neg        = floor(opts.batch_neg/numel(batch));
[im,rois,labels,btargets] = get_roi_batch(opts, imdb, k, batch);
nb = numel(labels);
nc = 2;
if opts.piecewise
% regression error only for positives
instance_weights = zeros(1,1,4*nc,nb,'single');
targets = zeros(1,1,4*nc,nb,'single');

for b=1:nb
  if labels(b)~=opts.bgLabel
    targets(1,1,4*(labels(b)-1)+1:4*labels(b),b) = btargets(b,:)';
    instance_weights(1,1,4*(labels(b)-1)+1:4*labels(b),b) = 1;
  end
end
end
rois = single(rois);

if opts.useGpu > 0
  im = gpuArray(im) ;
  rois = gpuArray(rois) ;
  targets = gpuArray(targets) ;
  instance_weights = gpuArray(instance_weights) ;
end
%if(nargout > 0 && opts.flip)
%    flip_idx = find(randi([0 1],size(boxes,1),1));
%    for i=flip_idx
%        im(:,:,:,i) = flip(im(:,:,:,i),2);
%    end
%end
%

if opts.piecewise
inputs = {'input', im, 'label', labels, 'rois', rois, 'targets', targets, ...
  'instance_weights', instance_weights} ;
else
inputs = {'input', im, 'label', labels, 'rois', rois} ;

end



% -------------------------------------------------------------------------
function imdb = mdnet_roi_setup_data(seqList, opts)
% -------------------------------------------------------------------------

imdb.images.name = {};
imdb.images.size = {};
imdb.images.set = {};
imdb.boxes.gtbox = {};
imdb.boxes.gtlabel = {};
%imdb.boxes.pgtidx = {};

imdb.boxes.pbox = {};
imdb.boxes.piou = {};
imdb.boxes.plabel = {};
for D = 1:length(seqList)
    
    dataset = seqList{D}.dataset;
    seqs_train = importdata(seqList{D}.list);
    
    for i = 1:length(seqs_train)
        seq = seqs_train{i};
        fprintf('sampling %s:%s ...\n', dataset, seq);
	opts.offset = length(imdb.images.name);
        
        config = genConfig(dataset, seq);
        imdb_ = roi_seq2roidb(config, opts);
    	imdb.images.name{i} = imdb_.images.name;
    	imdb.images.size{i} = imdb_.images.size;
    	imdb.images.set{i} =  imdb_.images.set;
    	imdb.boxes.gtbox{i} = imdb_.boxes.gtbox;
    	imdb.boxes.gtlabel{i} = imdb_.boxes.gtlabel;
    	imdb.boxes.pbox{i} = imdb_.boxes.pbox;
    	imdb.boxes.plabel{i} = imdb_.boxes.plabel;
    	imdb.boxes.piou{i} =imdb_.boxes.piou;
%        imdb.boxes.pgtidx = vertcat(imdb.boxes.pgtidx, imdb_.boxes.pgtidx);
    end
    imdb = add_bboxreg_targets(imdb);

    
end



% -------------------------------------------------------------------------
function [ net ] = mdnet_roi_init_train( opts )
% --------------------f-----------------------------------------------------
net = load(opts.netFile);
net = net.net;
net = dagnn.DagNN.loadobj(net);
pfc4 = find((arrayfun(@(a) strcmp(a.name, 'fc4f'), net.params)==1));
for i = 1:numel(net.params)
   if mod(i, 2) == 1
     net.params(i).weightDecay = 1;
     net.params(i).learningRate = 1;
  else
     net.params(i).weightDecay = 0;
     net.params(i).learningRate = 2;
  end
  
end
for i=pfc4:numel(net.params) - 4
  if mod(i-pfc4, 2) == 0
     net.params(i).weightDecay = 1;
     net.params(i).learningRate = 10;
  else
     net.params(i).weightDecay = 0;
     net.params(i).learningRate = 20;
  end
end

pFc6 = find(arrayfun(@(a) strcmp(a.name, 'predclsf'), net.params)==1);
% domain-specific layers
net.params(pFc6).value = 0.01 * randn(1,1,size(net.params(pFc6).value,3),2,'single');
net.params(pFc6+1).value = zeros(1, 2, 'single');

if opts.piecewise
ppredbbox = find(arrayfun(@(a) strcmp(a.name, 'predbboxf'), net.params)==1);
% domain-specific layers
net.params(ppredbbox).value = 0.01 * randn(1,1,size(net.params(ppredbbox).value,3),8,'single');
net.params(ppredbbox+1).value = zeros(1, 8, 'single');

end





% -------------------------------------------------------------------------
function genDir(path)
% -------------------------------------------------------------------------
if ~exist(path,'dir')
    mkdir(path);
end
        
function imdb = add_bboxreg_targets(imdb)
% add bbox regression targets
bgid = 1;
imdb.boxes.ptarget = cell(numel(imdb.images.name),1);

count = 1;
% add targets
for k=1:numel(imdb.images.name)
  for i = 1:numel(imdb.images.name{k})
  targets = zeros(numel(imdb.boxes.plabel{k}{i}),4);
  pos = (imdb.boxes.plabel{k}{i} ~= bgid);
  if isempty(pos)
    fprintf('no pos found (%d)\n',count);
    count = count + 1;
    continue;
  end

  ex_rois = imdb.boxes.pbox{k}{i}(pos,:) ;
  gt_rois = repmat(imdb.boxes.gtbox{k}{i}, sum(pos), 1);

  targets(pos,:) = bbox_transform(ex_rois, gt_rois);

  imdb.boxes.ptarget{k}{i} = targets;
  end
end

% compute means and stddevs
if ~isfield(imdb.boxes,'bboxMeanStd') || isempty(imdb.boxes.bboxMeanStd)

  sums = zeros(1,4);
  squared_sums = zeros(1,4);
  class_counts = eps;
  for k=1:numel(imdb.boxes.ptarget)
    for i=1:numel(imdb.boxes.ptarget{k})
      pos =  (imdb.boxes.plabel{k}{i}>0) ;
      labels = imdb.boxes.plabel{k}{i}(pos);
      targets = imdb.boxes.ptarget{k}{i}(pos,:);
      cls_inds = (labels~=bgid);
      if sum(cls_inds)>0
         class_counts = class_counts + sum(cls_inds);
         sums = sums + sum(targets(cls_inds,:));
         squared_sums = squared_sums + sum(targets(cls_inds,:).^2);
      end
    end
  end
  means = bsxfun(@rdivide,sums,class_counts);
  stds = sqrt(bsxfun(@rdivide,squared_sums,class_counts) - means.^2);

  imdb.boxes.bboxMeanStd{1} = means;
  imdb.boxes.bboxMeanStd{2} = stds;
  display('bbox target means:');
  display(means);
  display('bbox target stddevs:');
  display(stds);
else
  means = imdb.boxes.bboxMeanStd{1} ;
  stds = imdb.boxes.bboxMeanStd{2};
end

% normalize targets
for k=1:numel(imdb.boxes.ptarget)
    for i=1:numel(imdb.boxes.ptarget{k})
%   if imdb.images.set(i)<3
    pos = (imdb.boxes.plabel{k}{i} ~= bgid);
    labels = imdb.boxes.plabel{k}{i}(pos);
    targets = imdb.boxes.ptarget{k}{i}(pos,:);
      cls_inds = (labels~=bgid);
      if sum(cls_inds)>0
        targets(cls_inds,:) = bsxfun(@minus,targets(cls_inds,:),means);
        targets(cls_inds,:) = bsxfun(@rdivide,targets(cls_inds,:), stds);
      end
    imdb.boxes.ptarget{k}{i}(pos,:) = targets;
   end
end
