clear all
clc

% Input data Schizophrenia
cldata=xlsread('Schizophrenia.xlsx');
[nrow,ncol]=size(cldata);

% Split data into 70% train and 30% test
rng('default'); 
cv = cvpartition(size(cldata,1), 'HoldOut', 0.3);
idx = cv.test;
dataTrain = cldata(~idx,:);
dataTest = cldata(idx,:);

% Training data
X = dataTrain(:,1:ncol-1);
Y = dataTrain(:,ncol);
tic
gamma = 0.001;
SVMModel = fitcsvm(X,Y,'KernelFunction','rbf','KernelScale', gamma,'BoxConstraint',Inf);
CVSVMModel = crossval(SVMModel);

% Testing Data
XTest = dataTest(:,1:ncol-1);
YTest = dataTest(:,ncol);
label = predict(SVMModel,XTest);

% Counting the running time
timeElapsed=toc;
fprintf('Running Time: %i\n', timeElapsed);

% Check the accuracy
cp = classperf(YTest, label);
fprintf('Accuracy: %i\n', cp.CorrectRate*100);
