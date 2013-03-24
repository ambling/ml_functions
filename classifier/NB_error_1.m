function re=NB(dataset, alpha)

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

% indicator response matrix
Y = zeros(nSamples, nClasses);
for i=(1:size(trainGnd, 1)),
    Y(i, trainGnd(i)) = 1;
end

% Prior
Prior = sum(Y) ./ nSamples; % 1xC

% get a 3 dimensional matrix for each observation, feature, and class.
X_all = zeros(nSamples, nDims, nClasses);
for i = (1:nClasses),
    Y_i = Y(:, i);
    X_i = diag(Y_i) * trainFea;
    X_all(:,:,i) = X_i;   % NxDxC
    %X_i = X_i(any(X_i, 2), :);
end

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
calcY = zeros(size(testFea, 1), nClasses);
for i = (1:nClasses),
    % posterior
    Post = zeros(size(testFea,1), nDims);
    for j = (1:nDims),
        for k = (1:size(testFea, 1)),
            Post(k, j) = (sum(X_all(:,j,i)==testFea(k,j)) + alpha) ./ ...
                (sum(any(X_all(:,:,i), 2)) + alpha * nDims);
        end
    end
    calcY(:, i) = Prior(i) .* prod(Post, 2);
end

% stop timer of testing
disp('testing finished');
toc

[temp, calcLabels] = max(calcY, [], 2);
result = (testGnd ~= calcLabels);
rate = sum(result ~= 0) ./ (size(result, 1));
disp('error rate is:')
disp(rate)





