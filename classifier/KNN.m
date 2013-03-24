function re=KNN(dataset, k)

% re=KNN(dataset, k)
% this funtion that implements a classifier 
% of K-nearest neighbor model.
% 
% 'dataset' is used to choose the data set in folder './data/',
% with 1 indicates ORL database, 2 for USPS database 
% and 3 for Reuters21578, while others are unacceptable.
%
% 'k' is the parameter of the number of neighbors that taken into account.
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

% load training data 
load(trainFile);
trainFea = fea;   % SxD
trainGnd = gnd;   % SxC

% stop timer of training
disp('training finished');
toc


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%testing

% start timer of testing
tic

% load testing data 
load(testFile);
testFea = fea;   % NxD
testGnd = gnd;   % NxC
nTests = size(fea, 1); % N

% get distance matrix
% distance = dist(testFea, trainFea');   %NxS
% use inner product to get the squared distance
distance = bsxfun(@plus, sum(testFea.^2, 2), ...
    bsxfun(@plus, sum(trainFea'.^2), ...
    (-2).*(testFea * trainFea')));   %NxS


% get the k nearest neighbors using minK (from matlabtools)
[minDis, minIndex] = minK(distance, k);   % Nxk, Nxk

% get the labels of the neighbors
calcLabels = trainGnd(minIndex);  %Nxk
calcCountLabels = zeros(nTests, nClasses);  %NxC
for i = (1:nClasses),
    calcCountLabels(:, i) = sum(calcLabels == i, 2); %Nx1
end

% stop timer of testing
disp('testing finished');
toc


[temp, calcGnd] = max(calcCountLabels, [], 2);
result = (testGnd ~= calcGnd);
rate = sum(result ~= 0) ./ (size(result, 1));
disp('error rate is:')
disp(rate)





