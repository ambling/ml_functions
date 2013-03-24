function re=NB_inbuild(dataset, alpha)

% re=NB(dataset, alpha)
% this funtion that implements a classifier 
% of Naive Bayes model.
% 
% 'dataset' is used to choose the data set in folder './data/',
% with 1 indicates ORL database, 2 for USPS database 
% and 3 for Reuters21578, while others are unacceptable.
%
% 'alpha' is the smoothing parameter, with 0 for no smoothing,
% and use 1 as the common smoothing value
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
trainFea = fea;
trainGnd = gnd;
nSamples = size(trainFea, 1);
nDims = size(trainFea, 2);

nb = NaiveBayes.fit(trainFea, trainGnd, 'Distribution','mn');

% stop timer of training
disp('training finished');
toc


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%testing

% start timer of testing
tic


% load testing data 
load(testFile);
testFea = fea;
testGnd = gnd;

% get the testing label
calcLabels = predict(nb, testFea);

% stop timer of testing
disp('testing finished');
toc

result = (testGnd ~= calcLabels);
rate = sum(result ~= 0) ./ (size(result, 1));
disp('error rate is:')
disp(rate)





