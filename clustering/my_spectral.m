function [idx] = my_spectral(X, k, param)

% [idx] = my_spectral(X, k, param)
% the spectral clustering function that partitions X to k clusters.
% 
% X is the NxD matrix that contains N samples in which with D features.
% k is the number of clusters.
% param is set to 0 when unnormalized spectral clustering is used,
%      and to 1 when normalized one is used.
% 
% idx is the Nx1 index matrix of the clustering result.
%     
% learned and inspired from:
%    http://www.kyb.mpg.de/fileadmin/user_upload/files/
%         publications/attachments/Luxburg07_tutorial_4488%5b0%5d.pdf
% 
% my_kmeans.m is used to cluster the result.
%
% written by ambling<ambling07@gmail.com>, all rights reserved
% Mar 22st, 2013



%% Construct a full connected Gaussian similarity graph (weighted graph)
[nSamples, nDims] = size(X);

% get distance matrix
distance = bsxfun(@plus, sum(X.^2, 2), ...
    bsxfun(@plus, sum(X'.^2), (-2).*(X * X')));   % N x N

sigma = mean(mean(distance)); %use the mean of all distance as the parameter

% Gaussian similarity function
W = exp(-1.0 .* distance ./ (2.0 * sigma^2));


%% Compute the unnormalized Laplacian L
% get the degree matrix
D = diag(sum(W, 1));

% The unnormalized graph Laplacian matrix
L = D - W; % NxN


%% get the k smallest eigenvectors 
if param == 1,
    % normalized, to get generalized eigenvector
    [eigenvector, eigenvalue] = eig(L, D);
else
    % unnormalized
    [eigenvector, eigenvalue] = eig(L);
end
[sorted_engenvalue, sorted_index] = sort(diag(eigenvalue), 'ascend');
sorted_eigenvector = eigenvector(:, sorted_index);
U = sorted_eigenvector(:, (1:k));


%% use k-means to cluster the eigenvectors
idx = my_kmeans(U, k);

end