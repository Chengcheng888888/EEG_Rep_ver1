%% CSP feature extradtion 
%% = data pre-processing ===
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

%% Epoch extraction
% window for motor imagination 5s and relax 4.5s. For each trail
load ('D:\bci\ffhd.mat')
rang_M = [0 5.0] ;
rang_R = [0 4.5] ;

epochM = [];
sum_wnd_m = [];
for i = 1: size(flag_M, 1)
    wnd_m = rang_M * Fs +  flag_M(i); 
    epochM (:,:,i)= EEG.data(wnd_m(1):wnd_m(2)-1,1:8);
    sum_wnd_m = [sum_wnd_m;wnd_m];
end

epochR = [];
sum_wnd_r = [];
for i = 1: size(flag_R, 1)
    wnd_r = rang_R * Fs +  flag_R(i); 
    epochR(:,:,i )= EEG.data(wnd_r(1):wnd_r(2)-1,1:8);
    sum_wnd_r = [sum_wnd_r;wnd_r];
end
EEG.epoch = {epochM , epochR};
EEG.epoch_data = [epochM;epochR];



%% bandpass filter design
% Order =  3;
% [b,a]=butter(3,[8 35]/(500/2),'bandpass');
% EEG.EpoFltData = {};
% for k = 1:2 
%     for j=1:size(EEG.epoch{k},3)
%    tic;
%         EEG.EpoFltData{k}(:,:,j) = filtfilthd(b,a,EEG.epoch{k}(:,:,j)');
% %EEG.EpoFltData{k}(:,:,j)= filter(b,a,EEG.epoch{k}(:,:,j)');
%    toc
%       end
% end
%% reduce the edge effect. 2) construct a 4s window for each trail
rang_M_Edg = [0.5 4.5] ;
rang_R_Edg = [0.5 4.5] ;

%epochM_edg = [];
sum_wnd_m_edg = [];
for i = 1: size(flag_M, 1)
    wnd_m_edg = rang_M_Edg * Fs ;
    for j=1:size(EEG.EpoFltData{1},3)
    epochM_edg(:,:,j) = [EEG.EpoFltData{1}(1:8,wnd_m_edg(1)+1:wnd_m_edg(2),j)];
    end
    sum_wnd_m_edg = [sum_wnd_m_edg;wnd_m_edg];
end

% epochR_edg = [];
sum_wnd_r_edg = [];
for j = 1: size(flag_M, 1)
    
    wnd_r_edg = rang_R_Edg * Fs ;
    for j=1:size(EEG.EpoFltData{2},3)
    epochR_edg (:,:,j)= [EEG.EpoFltData{2}(1:8,wnd_r_edg(1)+1:wnd_r_edg(2),j)];
    end
    sum_wnd_r_edg = [sum_wnd_r_edg;wnd_r_edg];
end

EEG.epoch_edg = {epochM_edg , epochR_edg};
%EEG.epoch_data_edg = [epochM_edg;epochR_edg];
%% feature extraction  CSP
% extract the middle data 
traindata.x=EEG.epoch_edg{1}(:,:,1:round(0.5*(size(EEG.epoch_edg{1},3))));
traindata.y(1:round(0.5*(size(EEG.epoch_edg{1},3))))=ones(1,round(0.5*(size(EEG.epoch_edg{1},3))));
traindata.x(:,:,round(0.5*(size(EEG.epoch_edg{1},3)))+1:round(0.5*(size(EEG.epoch_edg{1},3)))+round(0.5*(size(EEG.epoch_edg{2},3))))=EEG.epoch_edg{2}(:,:,1:round(0.5*(size(EEG.epoch_edg{2},3))));
traindata.y(round(0.5*(size(EEG.epoch_edg{1},3)))+1:round(0.5*(size(EEG.epoch_edg{1},3)))+round(0.5*(size(EEG.epoch_edg{2},3))))=ones(1,round(0.5*(size(EEG.epoch_edg{2},3))))+1;

testdata.x=EEG.epoch_edg{1}(:,:,round(0.5*(size(EEG.epoch_edg{1},3)))+1:end);
testdata.y(1:size(testdata.x,3))=ones(1,size(testdata.x,3))
testdata2.x=EEG.epoch_edg{2}(:,:,round(0.5*(size(EEG.epoch_edg{2},3)))+1:end);
testdata.x(:,:,size(testdata.x,3)+1:size(testdata.x,3)+size(testdata2.x,3))=testdata2.x;
testdata.y(size(testdata.y,2)+1:size(testdata.y,2)+size(testdata2.x,3))=ones(1,size(testdata2.x,3))+1;
% C1_Data =  reshape(EEG.epoch_edg{1}(:,:,1:round(0.9*(size(EEG.epoch_edg{1},3)))),8,[]);
% C2_Data =  reshape(EEG.epoch_edg{2}(:,:,1:round(0.9*(size(EEG.epoch_edg{1},3)))),8,[]);

%[W] =  f_CSP(C1_Data,C2_Data);
[W] =  csp(traindata,'leavesingle');
%% epoch feature extraction  
Z = {};

class_label_m =  1;
class_label_r = -1;
label_data = [];

   for i=1:size(traindata.x,3)
        Z{i}= W * traindata.x(:,:,i);
  
         deno_M = var(Z{i}(1,:)) + var(Z{i}(2,:))+ var(Z{i}(7,:))+ var(Z{i}(8,:)) ;
        tr(1,i) = log(var(Z{i}(1,:))/deno_M);
         tr(2,i) = log(var(Z{i}(2,:))/deno_M);
         tr(3,i) = log(var(Z{i}(7,:))/deno_M);
         tr(4,i) = log(var(Z{i}(8,:))/deno_M); 
   end
         
clear Z
   for i=1:size(testdata.x,3)
        Z{i}= W * testdata.x(:,:,i);
  
         deno_M = var(Z{i}(1,:)) + var(Z{i}(2,:))+ var(Z{i}(7,:))+ var(Z{i}(8,:)) ;
       tst(1,i) = log(var(Z{i}(1,:))/deno_M);
         tst(2,i) = log(var(Z{i}(2,:))/deno_M);
         tst(3,i) = log(var(Z{i}(7,:))/deno_M);
         tst(4,i) = log(var(Z{i}(8,:))/deno_M); 
   end
      
%%
% taking 90% of the epoch data, from 1 to 0.9 * whole set ,
% same data set with spatial filter trainning data set.

%    test = (indices == i); train = ~test;  %产生测试集合训练集索引
    class = classify(tst',tr',traindata.y');
 %   classperf(cp,class,test)
Acc=1-mean(xor(class-1,testdata.y'-1));


