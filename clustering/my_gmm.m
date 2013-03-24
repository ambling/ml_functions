function [idx] = my_gmm(X, k)

% [idx] = my_gmm(X, k)
% the Gaussian Mixture Model clustering function 
%     that partitions X to k clusters.
% 
% X is the NxD matrix that contains N samples in which with D features.
% k is the number of clusters.
% 
% idx is the Nx1 index matrix of the clustering result.
%     
%
% learned and inspired by pluskid(http://blog.pluskid.org/?p=39)
% PCA method used to do the dimensionality reduction.
%
% written by ambling<ambling07@gmail.com>, all rights reserved
% Mar 22st, 2013

%% use PCA to do the dimensionality reduction
[coeff, score, latent] = pca(X);
X = score(:, latent>(latent(1)*0.3));

%% rand sample the centroids
cIdx = randsample(size(X, 1), k);
C = X(cIdx, :);

%% initialize parameters
[nSamples, nDims] = size(X);
mu = C;
prior = zeros(1, k);
sigma = zeros(nDims, nDims, k);

%get the distance matrix
D = bsxfun(@plus, sum(X.^2, 2), ...
    bsxfun(@plus, sum(C'.^2), (-2).*(X * C')));   % N x k

%label the samples
[in_exp, idx] = min(D, [], 2);

for i = (1:k)
    X_i = X(idx==i, :);
    prior(i) = size(X_i, 1)/nSamples;
    sigma(:, :, i) = cov(X_i);
end

%% iterating until converge
origL = -inf; %original 
threshold = 1e-15;
while true,
    %calculate gaussian probability density of multi-dimensional variable
    %http://en.wikipedia.org/wiki/Multivariate_normal_distribution
    prob = zeros(nSamples, k);
    for i = (1:k)
        X_shift = X - mu(i.*ones(nSamples, 1), :);  % X-mu
        in_exp = diag((X_shift / sigma(:,:,i)) * X_shift'); %Nx1
        coef = (2*pi)^(-nDims/2) / sqrt(det(sigma(:,:,i))); %1x1
        prob(:, i) = coef * exp(-0.5*in_exp); 
    end
    
    %%% EM method in ...
    % http://www.umiacs.umd.edu/~hal/courses/2011F_ML/out/gmm.pdf
    
    % compute expectation
    gamma = prob .* prior(ones(nSamples, 1), :);   % Nxk
    sum_gamma = sum(gamma, 2);   % Nx1
    gamma = gamma ./ sum_gamma(:, ones(k, 1));  % Nxk, Z_n,k
    
    % compute new value
    Nk = sum(gamma, 1);  %1xk, Z_k
    mu = diag(1./Nk) * gamma' * X;   %kxD
    prior = Nk/nSamples;   %1xk
    for i = (1:k)
        X_shift = X-mu(i.*ones(nSamples, 1), :);
        sigma(:, :, i) = (X_shift' * ...
            (diag(gamma(:, i)) * X_shift)) / Nk(i);
    end
    
    % check for convergence
    newL = sum(log(prob * prior'));
    if newL-origL < threshold,
        break;
    end
    origL = newL;
end

[temp, idx] = max(prob, [], 2);

end