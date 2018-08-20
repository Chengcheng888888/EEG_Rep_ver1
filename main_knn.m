%% CSP + KNN feature extradtion  
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
ex = 90;
ex1 = ex + 1;
C1_Data = EEG.epoch_edg{1}((ex*2000+1):(ex*2000+2000),1:8)' ;
C2_Data = EEG.epoch_edg{2}((ex*2000+1):(ex*2000+2000),1:8)' ;

[W] = f_CSP(C1_Data,C2_Data);
% spatial filtered singel trail signal Z
singel_trail = EEG.data(flag_M:(flag_R + 4.5*Fs),1:8)';
feature_trail = [C1_Data,C2_Data,EEG.epoch_edg{1}((ex1*2000+1):(ex1*2000+2000),1:8)'];
Z = W * feature_trail;
% plot  CSP filtered EEG data
figure ('color',[1 1 1])
ad_y = 0.5;
y = 1;
% for i = 1:size(Z,1)
for i = 1:2
%   subplot(size(Z,1),1,i)
    subplot(2,1,i)
    plot(Z(i,:))
%     set(gca, 'xtick',0:2000:5000,'ytick',[min(Z(i,:)) max(Z(i,:))]);
    set(gca, 'xtick',0:1000:5000);
    hold on 
    plot([2000,2000],[y*min(Z(i,:)) y*max(Z(i,:))],'r')
    hold on 
    plot([2000+2000,2000+2000],[min(Z(i,:)) max(Z(i,:))],'r')
%  figure labeling
    title ('band-pass filtered EEG after applying the CSP filters.')
    xlabel('Time in ms')
    ylabel('Amptitute')
    text(0,ad_y*max(Z(i,:)),'\leftarrow Movement','color','r')
    text(2000,ad_y*max(Z(i,:)),'\leftarrow Relax','color','r')
    text(4000,ad_y*max(Z(i,:)),'\leftarrow Movement','color','r')
    k = i;   
    legend(['Z',num2str(k)], 'Location','southwest') 
%    set(0,'defaultfigurecolor','w')
end 

% temp testing for Z7 and Z8
figure ('color',[1 1 1])
for i = 7:8
%   subplot(size(Z,1),1,i)
    subplot(2,1,2^(i-7))
    plot(Z(i,:))
%     set(gca, 'xtick',0:2000:5000,'ytick',[min(Z(i,:)) max(Z(i,:))]);
    set(gca, 'xtick',0:1000:5000);
    hold on 
    plot([2000,2000],[y*min(Z(i,:)) y*max(Z(i,:))],'r')
    hold on 
    plot([2000+2000,2000+2000],[min(Z(i,:)) max(Z(i,:))],'r')
%  figure labeling
    title ('band-pass filtered EEG after applying the CSP filters.')
    xlabel('Time in ms')
    ylabel('Amptitute')
    text(0,ad_y*max(Z(i,:)),'\leftarrow Movement','color','r')
    text(2000,ad_y*max(Z(i,:)),'\leftarrow Relax','color','r')
    text(4000,ad_y*max(Z(i,:)),'\leftarrow Movement','color','r')
    k = i;   
    legend(['Z',num2str(k)], 'Location','southwest') 
%    set(0,'defaultfigurecolor','w')
end 
% 3-4 5-6 
% temp testing for Z7 and Z8
figure ('color',[1 1 1])
for i = 3:4
%   subplot(size(Z,1),1,i)
    subplot(2,1,2^(i-3))
    plot(Z(i,:))
%     set(gca, 'xtick',0:2000:5000,'ytick',[min(Z(i,:)) max(Z(i,:))]);
    set(gca, 'xtick',0:1000:5000);
    hold on 
    plot([2000,2000],[y*min(Z(i,:)) y*max(Z(i,:))],'r')
    hold on 
    plot([2000+2000,2000+2000],[min(Z(i,:)) max(Z(i,:))],'r')
%  figure labeling
    title ('band-pass filtered EEG after applying the CSP filters.')
    xlabel('Time in ms')
    ylabel('Amptitute')
    text(0,ad_y*max(Z(i,:)),'\leftarrow Movement','color','r')
    text(2000,ad_y*max(Z(i,:)),'\leftarrow Relax','color','r')
    text(4000,ad_y*max(Z(i,:)),'\leftarrow Movement','color','r')
    k = i;   
    legend(['Z',num2str(k)], 'Location','southwest') 
%    set(0,'defaultfigurecolor','w')
end 

% temp testing for Z7 and Z8
figure ('color',[1 1 1])
for i = 5:6
%   subplot(size(Z,1),1,i)
    subplot(2,1,2^(i-5))
    plot(Z(i,:))
%     set(gca, 'xtick',0:2000:5000,'ytick',[min(Z(i,:)) max(Z(i,:))]);
    set(gca, 'xtick',0:1000:5000);
    hold on 
    plot([2000,2000],[y*min(Z(i,:)) y*max(Z(i,:))],'r')
    hold on 
    plot([2000+2000,2000+2000],[min(Z(i,:)) max(Z(i,:))],'r')
%  figure labeling
    title ('band-pass filtered EEG after applying the CSP filters.')
    xlabel('Time in ms')
    ylabel('Amptitute')
    text(0,ad_y*max(Z(i,:)),'\leftarrow Movement','color','r')
    text(2000,ad_y*max(Z(i,:)),'\leftarrow Relax','color','r')
    text(4000,ad_y*max(Z(i,:)),'\leftarrow Movement','color','r')
    k = i;   
    legend(['Z',num2str(k)], 'Location','southwest') 
%    set(0,'defaultfigurecolor','w')
end 
%% epoch feature extraction  
Z = {};
Z_sum = [];
class_label_m =  1;
class_label_r = -1;
label_data = [];
tic
for i = 1:size(sum_wnd_m_edg,1)
    for k = 1:2
        Z{k} = W * EEG.epoch_edg{k}(((i-1)*Fs*4+1):i*Fs*4,:)';
        Z_sum = [Z_sum,Z{k}];
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
     label_data = [label_data,class1,class2];
end
toc

%% KNN

% model for knn
Mdl = fitcknn(Train_X',Train_Y') ;
Mdl.NumNeighbors = 4;  
% prediction
flwrClass = predict(Mdl,Test_X);
% validation 
CVMdl = crossval(Mdl);  
% k-foler corss validatin lost caculation 
kloss = kfoldLoss(CVMdl);
rloss = resubLoss(Mdl)  ;


%==========================================================================
%#################### Training #################################
% Train the Classifiers on Training Data
size(Train_X)
disp('#######  Training The SVM Classsifier ##########')
TR_MDL.svm_mdls=svmtrain(Train_X,Train_Y,'showplot',true,'kktviolationlevel',0.05);
TR_MDL.svm_mdls
test = 8;

%% Perform 10-Fold Cross_validation
disp('########  Applying Cross-Validation    #################')
CASE='SVM';
[acc]=Cross_Validation_Haider(Train_Y', Train_X',CASE);
CV_acc=-acc.*100

%% Evaluation or Testing

[Label]=f_Adaptive_Learning_A(Test_X,TR_MDL);

%%  SVM Classification accuracy---------------------------------------------
SVM_class_error_Eval=(Test_Y-Label.SVM); %% The error from the classifier
SVM_ac=-(1-(sum((SVM_class_error_Eval).^2)./length(SVM_class_error_Eval)))*100
