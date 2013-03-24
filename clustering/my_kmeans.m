function [idx, C, sumD, D] = my_kmeans(X, k)

% [idx, C, sumD, D] = my_kmeans(X, k)
% the k-means clustering function that partitions X to k clusters.
% 
% X is the NxD matrix that contains N samples in which with D features.
% k is the number of clusters.
% 
% idx is the Nx1 index matrix of the clustering result.
% C is the kxD centroid matrix.
% sumD is the kx1 sum of distance between samples to centroids 
%     within each cluster.
% D is the Nxk matrix of distance between each sample to each centroid.
%     
% written by ambling<ambling07@gmail.com>, all rights reserved
% Mar 21st, 2013


%% rand sample the centroids
cIdx = randsample(size(X, 1), k);
C = X(cIdx, :);


%% calc new C from orig C until no changes are made
origC = zeros(k, size(X, 2));   % k x D
idx = zeros(size(X, 1), 1);
while(~all(all(origC == C))),
    origC = C;
    
    %get the distance matrix
    D = bsxfun(@plus, sum(X.^2, 2), ...
        bsxfun(@plus, sum(C'.^2), (-2).*(X * C')));   % N x k
    
    %label the samples
    [tmp, idx] = min(D, [], 2);
    
    %get the new centroids as the mean of the same labeled samples
    for i = (1:k),
        C(i, :) = mean(X(idx==i, :), 1);
    end
end

%% get the sumD
sumD = zeros(k, 1);
for i = (1:k)
    sumD(i) = sum(D(idx == i, i), 1);
end

end
