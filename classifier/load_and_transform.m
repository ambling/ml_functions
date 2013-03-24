function [augX, labels, Y] = load_and_transform(fileName, nClasses)

% augX, labels, Y = load_and_transform(fileName)
% load file with 'fileName' from datafiles,
% calculate augmented data 'augX', 
% and transformed indicator response 'matrixY',
% as well as the original labels
%
% 
% written by ambling<ambling07@gmail.com>, all rights reserved.

% load data file
load(fileName);

% augmented X for training data
nSamples = size(fea, 1);
augX = [ones(nSamples, 1), fea];

% indicator response matrix
Y = zeros(nSamples, nClasses);
for i=(1:size(gnd, 1)),
    Y(i, gnd(i)) = 1;
end

labels = gnd;