function mdnet_roi_prepare_model(varargin)
% MDNET_PREPARE_MODEL
% Prepare a initial CNN model for learning MDNet.
%
% conv1-3 are adopted from VGG-M network.
% fc4-fc6 are randomly initialized.
% fc6 will be replaced by multiple domain-specific layers when training MDNet.
%
% Hyeonseob Nam, 2015
% 

% conv1-3 layers from VGG-M network pretrained on ImageNet
opts.src_model = './models/imagenet-vgg-m-conv1-3.mat';
% output network 
opts.dst_model = './models/mdnet_roi_init_pw_2_2.mat';
opts.piecewise = 1;
opts = vl_argparse(opts, varargin);
if exist(opts.dst_model,'file')
    return;
end
    
%% load conv layers
load(opts.src_model);

new_layers = {};
for i=1:numel(layers)
    if strcmp(layers{i}.name,'conv4'), break; end
    switch (layers{i}.type)
        case 'conv'
            layers{i}.filters = layers{i}.weights{1};
            layers{i}.biases = layers{i}.weights{2};
            layers{i} = rmfield(layers{i},'weights');
            layers{i}.pad = 0;
            last_dim = size(layers{i}.biases,2);
        case 'pool'
            layers{i}.pad = 0;
    end
    new_layers{end+1} = layers{i};
end

%% init fc layers
scal = 1 ;
init_bias = 0.1;

%new_layers{end+1} = struct( 'type', 'roipool',...
%			    'name', 'roipool4',...
%			    'method', 'max',...
%			    'max','transform',1/16,...
%			    'subdivisions',[3,3],'flatten',0
%			);
%
% Block 4
new_layers{end+1} = struct('type', 'conv', ...
                           'name', 'fc4', ...
                           'filters', 0.01/scal * randn(5,5,last_dim,512,'single'),...
                           'biases', init_bias*ones(1,512,'single'), ...
                           'stride', 1, ...
                           'pad', 0);
new_layers{end+1} = struct('type', 'relu', 'name', 'relu4') ;
new_layers{end+1} = struct('type', 'dropout', 'name', 'drop4', 'rate', 0.5) ;

% Block 5
new_layers{end+1} = struct('type', 'conv', ...
                           'name', 'fc5', ...
                           'filters', 0.01/scal * randn(1,1,512,512,'single'),...
                           'biases', init_bias*ones(1,512,'single'), ...
                           'stride', 1, ...
                           'pad', 0) ;
new_layers{end+1} = struct('type', 'relu', 'name', 'relu5') ;
new_layers{end+1} = struct('type', 'dropout', 'name', 'drop5', 'rate', 0.5) ;

% Block 6
new_layers{end+1} = struct('type', 'conv', ...
                           'name', 'predcls', ...
                           'filters', 0.01/scal * randn(1,1,512,2,'single'), ...
                           'biases', zeros(1, 2, 'single'), ...
                           'stride', 1, ...
                           'pad', 0) ;

% Convert to DagNN.
net.layers = new_layers;
net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;

% Add ROIPooling layer.
pRelu3 = find(arrayfun(@(a) strcmp(a.name, 'relu3'), net.layers)==1);
if isempty(pRelu3)
    error('Cannot find last relu before fc');
end

net.addLayer('roipool', dagnn.ROIPooling('method','max','transform',1/16,...
    'subdivisions',[5,5],'flatten',0), ...
    {net.layers(pRelu3).outputs{1},'rois'}, 'xRP');

pFc4 = (arrayfun(@(a) strcmp(a.name, 'fc4'), net.layers)==1);
pRP = (arrayfun(@(a) strcmp(a.name, 'roipool'), net.layers)==1);
net.layers(pFc4).inputs{1} = net.layers(pRP).outputs{1};


% Add softmax loss layer.
ppredcls = (arrayfun(@(a) strcmp(a.name, 'predcls'), net.layers)==1);
net.addLayer('loss',dagnn.Loss(), ...
  {net.layers(ppredcls).outputs{1},'label'}, ...
  'losscls',{});

% Add bbox regression layer.
if opts.piecewise
  pfc6 = (arrayfun(@(a) strcmp(a.name, 'predclsf'), net.params)==1);
  pfc5 = (arrayfun(@(a) strcmp(a.name, 'drop5'), net.layers)==1);
  net.addLayer('predbbox',dagnn.Conv('size',[1 1 size(net.params(pfc6).value,3) 8],'hasBias', true), ...
    net.layers(pfc5).outputs{1},'predbbox',{'predbboxf','predbboxb'});

  net.params(end-1).value = 0.001 * randn(1,1,size(net.params(pfc6).value,3),8,'single');
  net.params(end).value = zeros(1,8,'single');

  net.addLayer('lossbbox',dagnn.LossSmoothL1(), ...
    {'predbbox','targets','instance_weights'}, ...
    'lossbbox',{});
end

net.rebuild();
net.meta.normalization.interpolation = 'bilinear';
clear layers;
save(opts.dst_model,'net');
