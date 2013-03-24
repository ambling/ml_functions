function [cost, sumD, D, idx] = my_get_cost(X, C)

%[cost, sumD, D, idx] = my_get_cost(X, C)
% get the cost and labels for kmeans and kmedoids.
% 
% X is NxD samples.
% C is kxD centroids or medoids
% 
% cost is the total cost.
% sumD is the kx1 sum of distance between samples to centroids 
%     within each cluster.
% D is the Nxk matrix of distance between each sample to each centroid.
% idx is the labels for each sample.
%     
% written by ambling<ambling07@gmail.com>, all rights reserved
% Mar 21st, 2013

%% get k
k = size(C, 1);

%% get the distance matrix
D = bsxfun(@plus, sum(X.^2, 2), ...
    bsxfun(@plus, sum(C'.^2), (-2).*(X * C')));   % N x k

%% label the samples
[tmp, idx] = min(D, [], 2);

%% get the sumD
sumD = zeros(k, 1);
for i = (1:k)
    sumD(i) = sum(D(idx == i, i), 1);
end

%% get cost
cost = sum(sumD);