function re=NB(dataset, method, alpha)

% re=NB(dataset, alpha)
% this funtion that implements a classifier 
% of Naive Bayes model.
% 
% 'dataset' is used to choose the data set in folder './data/',
% with 1 indicates ORL database, 2 for USPS database 
% and 3 for Reuters21578, while others are unacceptable.
%
% 'method' is the parameter to deal with probility, 
% with 0 for discrete counting,
% and 1 for normal distribution
%
% 'alpha' is the smoothing parameter, with 0 for no smoothing,
% and use 1 as the common smoothing value
%
% inspired and modified from Naive Bayes Classifier by Indraneel Biswas
% (http://www.mathworks.com/matlabcentral/fileexchange/
% 37737-naive-bayes-classifier/content/NaiveBayesClassifier.m)
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

% Prior
labels = unique(trainGnd);
Prior = zeros(1, nClasses);
for i = (1:nClasses),
    Prior(i) = (sum(double(trainGnd==labels(i))) + alpha) ./ ...
        (nSamples + alpha * nDims); % 1xC, with smoothing
end

if method == 1,
    % get mean and derivation of each class
    mu = zeros(nClasses, nDims);
    sigma = zeros(nClasses, nDims);
    for i = (1:nClasses),
        X_i = trainFea(trainGnd == labels(i), :);
        mu(i, :) = mean(X_i, 1);
        sigma(i, :) = std(X_i, 1);
    end
else
    likelihood = zeros(nClasses, nDims);
    evidence = ((sum(trainFea, 1) + alpha) ./ ...
        (nSamples + alpha * 2))';  % Dx1, with smoothing
    for i = (1:nClasses),
        X_i = trainFea(trainGnd == labels(i), :);
        likelihood(i, :) = (sum(X_i, 1) + alpha) ./ ...
            (size(X_i, 1) + alpha * nDims); % CxD, smoothing
    end
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
nTests = size(testFea, 1);

if method == 1,
    % get the testing label
    calcY = zeros(nTests, nClasses);
    for i = (1:nTests),
        probForAttr = normpdf(ones(nClasses, 1)*testFea(i,:), mu, sigma);
        calcY(i, :) = log(Prior) + sum(log(probForAttr), 2)';
    end   
    [temp, calcLabels] = max(calcY, [], 2);
    % disp(calcY);
else
    calcY = zeros(nTests, nClasses);
    for i = (1:nTests),
        index = testFea(i, :) ~= 0;
        post = prod(likelihood(:, index), 2) .* Prior' ...
            ./ prod(evidence(index), 1);
        calcY(i, :) = post';
    end
    [temp, calcLabels] = max(calcY, [], 2);
end

% stop timer of testing
disp('testing finished');
toc

result = (testGnd ~= labels(calcLabels));
rate = sum(result ~= 0) ./ (size(result, 1));
disp('error rate is:')
disp(rate)

