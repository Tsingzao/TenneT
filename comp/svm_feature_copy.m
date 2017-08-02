clear all;
clc;
addpath('../tSVD','../Liblinear','../util');

VidSize   = 32;
VidLength = 16;
VidFormat = 'gray';

file_name = 'penn';
nb_class  = '15';
pooling   = 'average';
%% Load UCF Demo Data

load(['../data/',file_name,'_train_',nb_class,'.mat']);
TrnData  = feature;
TrnLabel = label;
load(['../data/',file_name,'_test_',nb_class,'.mat']);
TestData = feature;
TestLabel= label;
clear feature label

%% Pooling
if strcmp(pooling,'average')
    featureTrain = mean(TrnData, 3);
    featureTest  = mean(TestData,3);
else
    featureTrain = max(TrnData, [], 3);
    featureTest  = max(TestData,[], 3);
end

%% 
fprintf('\n ====== Parameter Selection ======== \n')
tic;
best_C = parameter_selection(TrnLabel,sparse(featureTrain)');
ParameterSelection_Time = toc;

fprintf('\n ====== Training Linear SVM Classifier ======= \n')
tic;
models = train(TrnLabel, sparse(featureTrain)', ['-s 1 -c ',num2str(best_C),' -q']);
LinearSVM_TrnTime = toc;
clear featureTrain;

fprintf('\n ======= TentNet Testing ==========\n');
[xLabel_est, accuracy, decision_values] = predict(TestLabel, sparse(featureTest'), models, '-q');
Accuracy = sum(xLabel_est == TestLabel)/length(TestLabel);

%% Display Resluts
fprintf('\n ===== Results of TentNet, followed by a linear SVM classifier =====');
fprintf('\n     Parameter Selection time: %.2f secs.', ParameterSelection_Time);
fprintf('\n     Linear SVM training time: %.2f secs.', LinearSVM_TrnTime);
fprintf('\n     Testing Accuracy: %.2f%%', 100*Accuracy);