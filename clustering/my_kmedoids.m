function [idx, C, sumD, D] = my_kmedoids(X, k)

% [idx, C, sumD, D] = my_kmedoids(X, k)
% the k-my_kmedoids clustering function that partitions X to k clusters.
% 
% X is the NxD matrix that contains N samples in which with D features.
% k is the number of clusters.
% 
% idx is the Nx1 index matrix of the clustering result.
% C is the kxD medroid matrix.
% sumD is the kx1 sum of distance between samples to medoids 
%     within each cluster.
% D is the Nxk matrix of distance between each sample to each medroid.
%     
% written by ambling<ambling07@gmail.com>, all rights reserved
% Mar 21st, 2013


%% rand sample the medoids
cIdx = randsample(size(X, 1), k);
C = X(cIdx, :);


%% calc new C from orig C until no changes are made
origC = zeros(k, size(X, 2));   % k x D
idx = zeros(size(X, 1), 1);
while ~all(all(origC == C)),
    origC = C;
    
    [minCost, sumD, D, idx] = my_get_cost(X, C);
    
    %get the new medoids of the minCost
    for i = (1:k),
        X_i = X(idx==i, :);
        %iterate in the same cluster
        for mIdx = (1:size(X_i, 1)),
            medoid = X_i(mIdx, :);
            %only non-medoid
            if ~all(all(C(i,:)==medoid)),
               %get new cost
               newDist = bsxfun(@plus, sum(X_i.^2, 2), ...
                   bsxfun(@plus, sum(medoid'.^2),...
                   (-2).*(X_i * medoid')));   % N x 1
               newCost = sum(newDist);
               %if cost < minCost, then make the swap
               if newCost < sumD(i),
                   sumD(i) = newCost;
                   C(i,:) = medoid;
               end
            end
        end
    end
end


end
