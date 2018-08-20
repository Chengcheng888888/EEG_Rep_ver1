%% CSP feature extradtion 
%% = data pre-processing ===
clc; close all;clear all;
% loading Enobio raw EEG data  
% addpath = ('G:\desktop\BCI\ffd');
warning off
addpath(genpath('functions'))
data = load('20180808215944_MI_chengdan_MI01.easy');

cross_wait = 5;

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
% window for motor imagination 5s and relax 4.5s. For each epoch will eill
% add -0.5 seconds before action start and 0.5 seconds after action end
rang_M = [0.5 4.5] ;
rang_R = [0.5 4.0] ;

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

%% feature extraction  CSP
% extract the middle data 
ex = 10;
ex1 = ex + 1;
C1_Data = EEG.EpoFltData{1}((ex*2000+1):(ex*2000+2000),1:8)' ;
C2_Data = EEG.EpoFltData{2}((ex*1750+1):(ex*1750+1750),1:8)' ;

[W] = f_CSP(C1_Data,C2_Data);
% spatial filtered singel trail signal Z
singel_trail = EEG.data(flag_M:(flag_R + 4.5*Fs),1:8)';
feature_trail = [C1_Data,C2_Data,EEG.EpoFltData{1}((ex1*2000+1):(ex1*2000+2000),1:8)'];
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
    plot([2000+1750,2000+1750],[min(Z(i,:)) max(Z(i,:))],'r')
%  figure labeling
    title ('band-pass filtered EEG after applying the CSP filters.')
    xlabel('Time in ms')
    ylabel('Amptitute')
    text(0,ad_y*max(Z(i,:)),'\leftarrow Movement','color','r')
    text(2000,ad_y*max(Z(i,:)),'\leftarrow Relax','color','r')
    text(3750,ad_y*max(Z(i,:)),'\leftarrow Movement','color','r')
    k = i;   
    legend(['Z',num2str(k)], 'Location','southwest') 
%    set(0,'defaultfigurecolor','w')
end 

deno = var(Z(1,:)) + var(Z(2,:))+ var(Z(7,:))+ var(Z(8,:)) ;

 X_1 = log(var(Z(1,:))/deno);
 X_2 = log(var(Z(2,:))/deno);

 X_7 = log(var(Z(7,:))/deno);
 X_8 = log(var(Z(8,:))/deno); 


%% classification SVM 
% trainning data set construction 
X1 =  X{1};
X2 =  X{2};
Y1 =  -ones(1,size(X1,2));
Y2 =   ones(1,size(X2,2));
A = [X1,X2];
B = [Y1,Y2];
Train_X = A(:,1:1000:length(A)); 
Train_Y = B(:,1:1000:length(B));

Test_X = A(:,2:1000:length(A))'; 
Test_Y = B(:,2:1000:length(B))';

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
