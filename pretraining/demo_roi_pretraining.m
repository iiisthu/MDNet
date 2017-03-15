%% DEMO_PRETRAINING
%
% Training MDNet models.
%
% Hyeonseob Nam, 2015
%

clear;
close all;
clc;
run(fullfile(fileparts(mfilename('fullpath')), ...
  '..','matconvnet','matlab', 'vl_setupnn.m'));
addpath('pretraining');
addpath('tracking');
addpath('utils');
addpath('pretraining/bbox_functions');
% Prepare a CNN model for learning MDNet.
mdnet_roi_prepare_model;

%% Training MDNet using the sequences from {VOT13,14,15}-{OTB100}
% for experiments on OTB
% mdnet_roi_pretrain('seqsList',...
%     {struct('dataset','vot2013','list','pretraining/seqList/vot13-otb.txt'),...
%     struct('dataset','vot2014','list','pretraining/seqList/vot14-otb.txt'),...
%     struct('dataset','vot2015','list','pretraining/seqList/vot15-otb.txt')},...
%     'outFile', fullfile('models','mdnet_vot-otb_new.mat'),...
%     'roiDir', fullfile('models','data_vot-otb'));

%% Training MDNet using the sequences from {OTB}-{VOT14}
% for experiments on VOT14
mdnet_roi_pretrain('seqsList',...
    {struct('dataset','otb','list','pretraining/seqList/otb-vot14.txt')},...
    'outFile', fullfile('models','mdnet_otb-vot14_new.mat'),...
    'imdbDir', fullfile('models','data_otb-vot14'));

%% Training MDNet using the sequences from {OTB}-{VOT15}
% for experiments on VOT15
mdnet_roi_pretrain('seqsList',...
    {struct('dataset','otb','list','pretraining/seqList/otb-vot15.txt')},...
    'outFile', fullfile('models','mdnet_otb-vot15_new.mat'),...
    'imdbDir', fullfile('models','data_otb-vot15'));
