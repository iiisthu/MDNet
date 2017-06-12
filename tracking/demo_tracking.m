%% DEMO_TRACKING
%
% Running the MDNet tracker on a given sequence.
%
% Hyeonseob Nam, 2015
%

clear;

conf = genConfig('otb','Basketball');
% conf = genConfig('vot2015','ball1');

switch(conf.dataset)
    case 'otb'
        net = fullfile('models','mdnet_vot-otb_mytrain.mat');
    case 'vot2014'
        net = fullfile('models','mdnet_otb-vot14_mytrain.mat');
    case 'vot2015'
        net = fullfile('models','mdnet_otb-vot15_mytrain.mat');
end

ts_table = [{'dataset'},{'mdnet_init'},{'bbox_training'}, {'generate_examples'},...
    {'mdnet_convX'},{'mdnet_finetune_fc'},{'prepare_onlupd'}, {'ou_draw_candidates'}, ...
    {'ou_cand_convX'},{'ou_eva_candidates'}, {'ou_bbox_prediction'}, ...
    {'ou_store_sample'},{'ou_store_conv'} ...
    {'ou_update'},{'ou_total'}, {'ou_total_expel_update'}
	];
[result, ts] = mdnet_run(conf.imgList, conf.gt(1,:), net, true);
ts_table = [ts_table; [{'Bolt'}, ts]];
% write timestamp to csv data
[N,l] = size(ts_table);
fid = fopen('time_statistics.csv', 'w') ;
for i = 1:N
fprintf(fid, '%s,', ts_table{i,1:end-1}) ;
fprintf(fid, '%s\n', ts_table{i,end}) ;
end
fclose(fid) ;
