%% CSP feature extradtion 
%% = data pre-processing ===
clc; close all;clear all;
% loading Enobio raw EEG data  
% addpath = ('G:\desktop\BCI\ffd');
warning off
addpath(genpath('functions'))
data = load('20180808215944_MI_chengdan_MI01.easy');

epoch_range = [0.5 4.5];
time_ranges = [0 5];
lambda = 0.1;
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
for i = 1: size(flag_M, 1)
    wnd_m = rang_M * Fs +  flag_M(i); 
    epochM = [epochM;EEG.data(wnd_m(1):wnd_m(2),1:8)];
end

epochR = [];
for i = 1: size(flag_R, 1)
    wnd_r = rang_R * Fs +  flag_R(i); 
    epochR = [epochR;EEG.data(wnd_r(1):wnd_r(2),1:8)];
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
C1_Data = EEG.EpoFltData{1}(250:2250,1:8)' ;
C2_Data = EEG.EpoFltData{2}(250:2000,1:8)' ;

[W] = f_CSP(C1_Data,C2_Data);
% spatial filtered singel trail signal Z
singel_trail = EEG.data(flag_M:(flag_R + 4.5*Fs),1:8)';
singel_trail = [EEG.EpoFltData{1}(1:3000,1:8)',EEG.EpoFltData{1}(1:2750,1:8)'];
Z = W * singel_trail;
% plot  CSP filtered EEG data
 figure 
 subplot(4,1,1)
 plot(Z(1,:))
 
 subplot(4,1,2)
 plot(Z(2,:))
 
 subplot(4,1,3)
 plot(Z(7,:))
 
 subplot(4,1,4)
 plot(Z(8,:))

% log-variance feature extraction
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
