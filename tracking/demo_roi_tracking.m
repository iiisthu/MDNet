%% DEMO_TRACKING
%
% Running the MDNet tracker on a given sequence.
%
% Hyeonseob Nam, 2015
%

clear; close all;

run(fullfile(fileparts(mfilename('fullpath')), ...
  '..','matconvnet','matlab', 'vl_setupnn.m'));
addpath('pretraining');
addpath('tracking');
addpath('utils');
addpath('pretraining/bbox_functions');
%conf = genConfig('otb','Matrix');
conf = genConfig('vot2014','fish1');

switch(conf.dataset)
    case 'otb'
        %net = fullfile('models','net-deployed_shared_bbox_relu_sub7_fixed_test-epoch11.mat');
        %net = fullfile('models','net-deployed_sub3_oneas_large.mat');
        %net = fullfile('models', 'net-deployed_sub3_oneas_large_nopw_epoch3');
        %net = fullfile('models','net-deployed_vot-otb_sub3_oneas_large_nopw_epoch4.mat');
        %net = fullfile('models', 'net-deployed_shared_bbox_relu_new_start.mat');
		%net = fullfile('models', 'net-deployed_otb-vot14_sub3_oneas_large_nopw_validate2-epoch4.mat');
		 %net = fullfile('models', 'net-deployed_otb-vot14_sub3_oneas_large_nopw-epoch7.mat')
		%net = fullfile('models', 'net-deployed_vot-otb_sub3_oneas_large_nopw_bothboundary_flip-epoch6.mat');
        net = fullfile('models', 'net-deployed_vot-otb_start_epoch13.mat');
    case 'vot2014'
        net = fullfile('models', 'net-deployed_sub3_oneas_large_nopw_epoch3');

        %net = fullfile('models','mdnet_roi_otb-vot14.mat');
    case 'vot2015'

        %net = fullfile('models', 'net-deployed_vot-otb_start_epoch13.mat');
        %net = fullfile('models','mdnet_roi_otb-vot15.mat');
end

%ts_table = [{'dataset'},{'mdnet_init'},{'bbox_training'}, {'generate_examples'},...
%    {'mdnet_convX'},{'mdnet_finetune_fc'},{'prepare_onlupd'}, {'ou_draw_candidates'}, ...
%    {'ou_cand_convX'},{'ou_eva_candidates'}, {'ou_bbox_prediction'}, ...
%    {'ou_store_sample'},{'ou_store_conv'} ...
%    {'ou_update'},{'ou_total'}, {'ou_total_expel_update'}
%	];
[result, ts] = mdnet_roi_run(conf.imgList, conf.gt(1,:), net, true);
%ts_table = [ts_table; [{'Diving'}, ts]];
%% write timestamp to csv data
%[N,l] = size(ts_table);
%fid = fopen('time_statistics.csv', 'w') ;
%for i = 1:N
%fprintf(fid, '%s,', ts_table{i,1:end-1}) ;
%fprintf(fid, '%s\n', ts_table{i,end}) ;
%end
%fclose(fid) ;
