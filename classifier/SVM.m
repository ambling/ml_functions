function re=SVM(dataset, trainer, options)

% re=SVM(dataset, trainer, options)
% this funtion that implements a classifier of SVM model 
% with the help of libsvm(http://www.csie.ntu.edu.tw/~cjlin/libsvm).
% 
% 'dataset' is used to choose the data set in folder './data/',
% with 1 indicates ORL database, 2 for USPS database 
% and 3 for Reuters21578, while others are unacceptable.
%
% 'trainer' is the trainer to use 
% with 1 for libsvm and 2 for liblinear
%
% 'options' is the A string of training options 
% in the same format as that of LIBSVM.
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


%% add path of libsvm or liblinear
if trainer == 1,
    path('./libsvm', path);
else
    path('./liblinear', path);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% training

% start timer of training
tic

% load training data 
load(trainFile);
trainFea = fea;   % SxD
trainGnd = gnd;   % SxC

% train the svm
if trainer == 1,
    model = svmtrain(trainGnd, trainFea, options);
elseif trainer == 2,
    model = train(trainGnd, sparse(trainFea));
end


% stop timer of training
disp('training finished');
toc


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% testing

% start timer of testing
tic

% load testing data 
load(testFile);
testFea = fea;   % NxD
testGnd = gnd;   % NxC
nTests = size(fea, 1); % N

% predict the data
if trainer == 1,
    svmpredict(testGnd, testFea, model);
elseif trainer == 2,
    predict(testGnd, sparse(testFea), model);
end

% stop timer of testing
disp('testing finished');
toc





