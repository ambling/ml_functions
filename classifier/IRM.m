function re=IRM(dataset, lambda)

% re=IRM(dataset, lamda)
% this funtion that implements a classifier 
% of Indicator Response Matrix model.
% 
% 'dataset' is used to choose the data set in folder './data/',
% with 1 indicates ORL database, 2 for USPS database 
% and 3 for Reuters21578, while others are unacceptable.
%
% 'lamda' is the regularization parameter
%
% written by ambling<ambling07@gmail.com>, all rights reserved.

if dataset == 1,
    trainFile = './data/ORL_train.mat';
    testFile = './data/ORL_test.mat';
    nClasses = 40; %40 classes in training data
elseif dataset == 2,
    trainFile = './data/USPS_train.mat';
    testFile = './data/USPS_test.mat';
    nClasses = 10; %10 classes in training data
elseif dataset == 3,
    trainFile = './data/Reuters_train.mat';
    testFile = './data/Reuters_test.mat';
    nClasses = 40; %40 classes in training data
else
    re='Error using dataset: 1 indicates ORL database, ';
    re = [re, '2 for USPS database and 3 for Reuters21578, '];
    re = [re, 'others are unacceptable'];
    return;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%training

% start timer of training
tic

% load training data and transfrom it
[augX, labels, Y] = load_and_transform(trainFile, nClasses);

% coefficient matrix
B = (augX' * augX + lambda * eye(size(augX, 2))) \ augX' * Y;

% stop timer of training
disp('training finished');
toc


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%testing

% start timer of testing
tic

% load testing data and transform it
[testAugX, testLabels, testY] = load_and_transform(testFile, nClasses);

% calculate the result matrix
calcY = testAugX * B;

% stop timer of testing
disp('testing finished');
toc


[temp, calcLabels] = max(calcY, [], 2);
result = (testLabels ~= calcLabels);
rate = sum(result ~= 0) ./ (size(result, 1));
disp('error rate is:')
disp(rate)





