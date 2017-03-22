function [ net, opts] = mdnet_roi_init(image, net)
% MDNET_INIT
% Initialize MDNet tracker.
%
% Hyeonseob Nam, 2015
% 

%% set opts
% use gpu
opts.useGpu = true;

% model def
opts.net_file = net;

% test policy
opts.batchSize_test = 256; % <- reduce it in case of out of gpu memory

% bounding box regression
opts.bbreg = true;
opts.bbreg_nSamples = 1000;

% learning policy
opts.batchSize = 128;
opts.batch_pos = 32;
opts.batch_neg = 96;

% initial training policy
opts.learningRate_init = 0.01 * [ones(30,1); 0.1*ones(70,1);0.01*ones(100,1); 0.001*ones(100,1)]; % x10 for fc6
opts.maxiter_init = 300;

opts.nPos_init = 500;
opts.nNeg_init = 5000;
opts.posThr_init = 0.7;
opts.negThr_init = 0.5;

% update policy
opts.learningRate_update = 0.005; % x10 for fc6
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
opts.minIn = 107;
opts.nmsThreshold = 0.3 ;
opts.confThreshold = 0.5 ;
opts.visualize = true;
% scaling policy
opts.scale_factor = 1.05;

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

pfc8 = net.getLayerIndex('predcls') ;
net.addLayer('probcls',dagnn.SoftMax(),net.layers(pfc8).outputs{1},...
  'probcls',{})
net.vars(net.getVarIndex('probcls')).precious = 1 ;
net.vars(net.getVarIndex('predbbox')).precious = 1 ;
