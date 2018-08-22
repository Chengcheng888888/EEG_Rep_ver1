%% CSP feature extradtion 
%% = data pre-processing ===
tic;
clc; close all;clear all;
% loading Enobio raw EEG data  
% addpath = ('G:\desktop\BCI\ffd');
warning off
addpath(genpath('functions'))
%% data loading part I
% data0 = load('D:\bci\09chendan\20180808214850_MI_chengdan_act01.easy');
% data1 = load('D:\bci\\09chendan\20180808215944_MI_chengdan_MI01.easy');
% data2 = load('D:\bci\09chendan\20180808221159_MI_chengdan_mi02.easy');
% data3 = load('D:\bci\09chendan\20180808223920_MI_chengdan_mi03.easy');
% data4 = load('D:\bci\09chendan\20180808225428_MI_chengdan_mi05.easy');
% data5 = load('D:\bci\09chendan\20180808230805_MI_chengdan_mi06.easy');
% data6 = load('D:\bci\09chendan\20180808231759_MI_chengdan_mi078.easy');
% data7 = load('D:\bci\09chendan\20180808233449_MI_chengdan_mi09.easy');
%  
% data_all = [data1;data2;data3;data4;data5;data6;data7];
%% data loading part II

 data1 = load ('D:\bci\09.mat');
 data = data1.data_all;
%%
cross_wait = 5.0;
relax_wait = 4.5;

% EEG structure define
EEG.srate = 500;
Fs = EEG.srate ;
EEG.data = data;
EEG.epoch = [];
EEG.filename = 'Motor Imagery' ;

% find  markers
flag_M = find(data(:,9) ~= 0 );
flag_R = flag_M + cross_wait * EEG.srate ;

for e = 1:size(flag_M,1)
        EEG.event(e).latency = flag_M(e);
        EEG.event(e).duration = 5;
        EEG.event(e).type = 'M';
end
fsz = size(flag_M,1);
for r = 1:size(flag_R,1)
        EEG.event(fsz(1) + r).latency = flag_R(r);
        EEG.event(fsz(1) + r).duration = 4.5;
        EEG.event(fsz(1) + r).type = 'R';
end
%% Epoch extraction
% window for motor imagination 5s and relax 4.5s. For each trail
rang_M = [0 5.0] ;
rang_R = [0 4.5] ;

epochM = [];
sum_wnd_m = [];
for i = 1: size(flag_M, 1)
    wnd_m = rang_M * Fs +  flag_M(i); 
    epochM = [epochM;EEG.data(wnd_m(1):wnd_m(2)-1,1:8)];
    sum_wnd_m = [sum_wnd_m;wnd_m];
end

epochR = [];
sum_wnd_r = [];
for i = 1: size(flag_R, 1)
    wnd_r = rang_R * Fs +  flag_R(i); 
    epochR = [epochR;EEG.data(wnd_r(1):wnd_r(2)-1,1:8)];
    sum_wnd_r = [sum_wnd_r;wnd_r];
end
EEG.epoch = {epochM , epochR};
EEG.epoch_data = [epochM;epochR];

%% bandpass filter design
Order =  3;
[b,a]=butter(3,[2 35]/(500/2),'bandpass');
EEG.EpoFltData = {};
for k = 1:2 
EEG.EpoFltData{k} = filtfilthd(b,a,EEG.epoch{k});
end
%% reduce the edge effect. 2) construct a 4s window for each trail
rang_M_Edg = [0.5 4.5] ;
rang_R_Edg = [0.5 4.5] ;

epochM_edg = [];
sum_wnd_m_edg = [];
for i = 1: size(flag_M, 1)
    wnd_m_edg = rang_M_Edg * Fs + (i-1)* Fs*cross_wait;
    epochM_edg = [epochM_edg;EEG.EpoFltData{1}(wnd_m_edg(1):wnd_m_edg(2)-1,1:8)];
    sum_wnd_m_edg = [sum_wnd_m_edg;wnd_m_edg];
end

epochR_edg = [];
sum_wnd_r_edg = [];
for j = 1: size(flag_M, 1)
    wnd_r_edg = rang_R_Edg * Fs + (j-1)* Fs*relax_wait;;
    epochR_edg = [epochR_edg;EEG.EpoFltData{2}(wnd_r_edg(1):wnd_r_edg(2)-1,1:8)];
    sum_wnd_r_edg = [sum_wnd_r_edg;wnd_r_edg];
end

EEG.epoch_edg = {epochM_edg , epochR_edg};
EEG.epoch_data_edg = [epochM_edg;epochR_edg];

%% feature extraction  CSP
% extract the middle data 
Sum_W = {};
for i = 1:size(flag_M, 1)
    C1_Data = EEG.epoch_edg{1}(((i-1)*2000+1):((i-1)*2000+2000),1:8)' ;
    C2_Data = EEG.epoch_edg{2}(((i-1)*2000+1):((i-1)*2000+2000),1:8)' ;
    [W] =  f_CSP(C1_Data,C2_Data);
    Sum_W{i} = W;
end 
%% epoch feature extraction  
tic;
Z = {};
Z_sum = [];
class_label_m =  1;
class_label_r = -1;
label_data = [];

features_cell ={};
for iw = 1:size(flag_M, 1)
 
    for i = 1:size(sum_wnd_m_edg,1)
        for k = 1:2
            Z{k} = Sum_W{iw} * EEG.epoch_edg{k}(((i-1)*Fs*4+1):i*Fs*4,:)';
%             Z_sum = [Z_sum,Z{k}];
        end
        deno_M = var(Z{1}(1,:)) + var(Z{1}(2,:))+ var(Z{1}(7,:))+ var(Z{1}(8,:)) ;
         M_1 = log(var(Z{1}(1,:))/deno_M);
         M_2 = log(var(Z{1}(2,:))/deno_M);
         M_7 = log(var(Z{1}(7,:))/deno_M);
         M_8 = log(var(Z{1}(8,:))/deno_M); 

         deno_R = var(Z{2}(1,:)) + var(Z{2}(2,:))+ var(Z{2}(7,:))+ var(Z{2}(8,:)) ;
         R_1 = log(var(Z{2}(1,:))/deno_M);
         R_2 = log(var(Z{2}(2,:))/deno_M);
         R_7 = log(var(Z{2}(7,:))/deno_M);
         R_8 = log(var(Z{2}(8,:))/deno_M); 


         class1 = [M_1;M_2;M_7;M_8;class_label_m];
         class2 = [R_1;R_2;R_7;R_8;class_label_r];
%          label_data = [label_data,class1,class2];
      features_cell.data{2*i-1  ,iw} = class1(1:4,:); 
      features_cell.data{2*i    ,iw} = class2(1:4,:);
      features_cell.label{2*i-1  ,iw} = class1(5,:); 
      features_cell.label{2*i    ,iw} = class2(5,:);
      
    end

  
end
  toc;
%% spatial filtered singel trail signal Z
% feature selection 
%  features_M = reshape(cell2mat(features_cell),5,[]);
%  features = features_M(1:4,:)';
%  features_slt= reshape(features_cell,448,1);
%  f_slc_col = W;
% features = 
features = features_cell.data
labels = features_cell.data{:,1} +2 ;

[F_MI,W_MI] = MI(features,labels,3);

%% Test KNN classification accuracy with the different feature sets using the same data points for training and testing
% (note that k = 1 always leads to 100% accuracy without an independent test set). 
test = 8;
k = 5;     % k used in KNN classification

hypos_MI = KNN(features(:,F_MI(1:4)),features(:,F_MI(1:4)),labels,k);
fprintf('Best 10 features from MI: %0.2f%% correct.\n',sum(hypos_MI == labels)/length(labels)*100);
test = 8;
