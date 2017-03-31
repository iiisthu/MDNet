function [ net_conv, net_fc, opts] = mdnet_roi_init(image, net)
% MDNET_INIT
% Initialize MDNet tracker.
%
% Hyeonseob Nam, 2015
% 

%% set opts
% use gpu
opts.gpus = [3];

% model def
opts.net_file = net;

% test policy
opts.batchSize_test = 256; % <- reduce it in case of out of gpu memory

% bounding box regression

% learning policy
opts.batchSize = 128;
opts.batch_pos = 32;
opts.batch_neg = 96;

% initial training policy
opts.maxiter_init = 200;
opts.learningRate_init = 0.0005 * [ones(opts.maxiter_init/2,1);0.1*ones(opts.maxiter_init/2,1)]; % x10 for fc6

opts.nPos_init = 500;
opts.nNeg_init = 5000;
opts.posThr_init = 0.7;
opts.negThr_init = 0.5;

% update policy
opts.learningRate_update = 0.0005; % x10 for fc6
opts.maxiter_update = 10;

opts.nPos_update = 50;
opts.nNeg_update = 200;
opts.posThr_update = 0.7;
opts.negThr_update = 0.3;

opts.update_interval = 10; % interval for long-term update

% data gathering policy
opts.nFrames_long = 100; % long-term period
opts.nFrames_short = 20; % short-term period

% cropping policy
opts.crop_size = 107;
opts.crop_mode = 'wrap';
opts.crop_padding = 16;
opts.maxIn = 400;
opts.minIn = 200;
opts.nmsThreshold = 0.3 ;
opts.confThreshold = 0.5 ;
opts.visualize = false;
% scaling policy
opts.scale_factor = 1.05;
opts.piecewise = 1;
if opts.piecewise
opts.derOutputs = {'losscls', 1, 'lossbbox', 0.2};
else
opts.derOutputs = {'losscls', 1};
end
% sampling policy
opts.nSamples = 256;
opts.trans_f = 0.6; % translation std: mean(width,height)*trans_f/2
opts.scale_f = 1; % scaling std: scale_factor^(scale_f/2)

% set image size
opts.imgSize = size(image);

%% load net
net = load(opts.net_file);
if isfield(net,'net'), net = net.net; end
% Load the network and put it in test mode.
net = dagnn.DagNN.loadobj(net);
net.mode = 'test' ;

% Mark class and bounding box predictions as `precious` so they are
% not optimized away during evaluation.

net.vars(net.getVarIndex('probcls')).precious = 1 ;
net.vars(net.getVarIndex('predbbox')).precious = 1 ;

pFc6 = find(arrayfun(@(a) strcmp(a.name, 'predclsf'), net.params)==1);
for i = 1:pFc6 - 1
   if mod(i, 2) == 1
     net.params(i).weightDecay = 1;
     net.params(i).learningRate = 1;
  else
     net.params(i).weightDecay = 0;
     net.params(i).learningRate = 2;
  end

end
for i=pFc6:numel(net.params) - 2 
  if mod(i-pFc6, 2) == 0
     net.params(i).weightDecay = 1;
     net.params(i).learningRate = 10;
  else
     net.params(i).weightDecay = 0;
     net.params(i).learningRate = 20;
  end
end

lrelu3 = find(arrayfun(@(a) strcmp(a.name, 'relu3'), net.layers)==1);
vx10 = find(arrayfun(@(a) strcmp(a.name, 'x10'), net.vars) == 1);
pconv3b = find(arrayfun(@(a) strcmp(a.name, 'conv3b'), net.params) == 1);
net_fc = copy(net);
net_conv = copy(net);

for i = 1:numel(net.layers)
if i <= lrelu3
net_fc.removeLayer(net.layers(i).name);
else
net_conv.removeLayer(net.layers(i).name);
end
end
net_conv.vars = net.vars(1:vx10);
net_fc.vars = net.vars(vx10:end);
net_conv.params = net.params(1:pconv3b);
net_fc.params = net.params(pconv3b+1:end);

net_conv = dagnn.DagNN.loadobj(net_conv);
net_fc = dagnn.DagNN.loadobj(net_fc);
net_conv.rebuild();
net_fc.rebuild();
if numel(opts.gpus) > 0
net_conv.move('gpu');
net_fc.move('gpu');

end
