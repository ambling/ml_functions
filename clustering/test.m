%% introduction
% this file is used to run the test of each algo.
% 
% first deal with the data, then call the function and resport the result.
%

%% load evaluation functions
path('./EvaluateMetric', path);

%% kmeans
for nCluster = [5,7],
   disp(['Cluster number is: ', int2str(nCluster)]);
   for file = (1:10),
       filename=['MNIST/',int2str(nCluster),...
           'Class/',int2str(file),'.mat'];
       disp(['Begin testing file: ',filename]);      
       load('MNIST/MNIST.mat');
       load(filename);
       fea = fea(sampleIdx,:);
       gnd = gnd(sampleIdx,:);
       fea(:,zeroIdx) = [];
       tic;
       [idx, C, sumD, D] = my_kmeans(fea, nCluster);
       toc;
       idx = bestMap(gnd, idx);
       accuracy = length(find(gnd == idx))/length(gnd);
       disp(['The accuracy is: ', num2str(accuracy * 100), '%']);
       MIhat = MutualInfo(gnd,idx);
       disp(['The MIhat is: ', num2str(MIhat)]);
   end
end

disp('Cluster number is: 10');
disp('Begin testing file: MNIST/MNIST.mat');
load('MNIST/MNIST.mat');
tic;
[idx, C, sumD, D] = my_kmeans(fea, 10);
toc;
idx = bestMap(gnd, idx);
accuracy = length(find(gnd == idx))/length(gnd);
disp(['The accuracy is: ', num2str(accuracy * 100), '%']);
MIhat = MutualInfo(gnd,idx);
disp(['The MIhat is: ', num2str(MIhat)]);

for nCluster = [5,10],
   disp(['Cluster number is: ', int2str(nCluster)]);
   for file = (1:10),
       filename=['COIL20/',int2str(nCluster),...
           'Class/',int2str(file),'.mat'];
       disp(['Begin testing file: ',filename]);
       load('COIL20/COIL20.mat');
       load(filename);
       fea = fea(sampleIdx,:);
       gnd = gnd(sampleIdx,:);
       fea(:,zeroIdx) = [];
       tic;
       [idx, C, sumD, D] = my_kmeans(fea, nCluster);
       toc;
       idx = bestMap(gnd, idx);
       accuracy = length(find(gnd == idx))/length(gnd);
       disp(['The accuracy is: ', num2str(accuracy * 100), '%']);
       MIhat = MutualInfo(gnd,idx);
       disp(['The MIhat is: ', num2str(MIhat)]);
   end
end

disp('Cluster number is: 20');
disp('Begin testing file: COIL20/COIL20.mat');
load('COIL20/COIL20.mat');
tic;
[idx, C, sumD, D] = my_kmeans(fea, 20);
toc;
idx = bestMap(gnd, idx);
accuracy = length(find(gnd == idx))/length(gnd);
disp(['The accuracy is: ', num2str(accuracy * 100), '%']);
MIhat = MutualInfo(gnd,idx);
disp(['The MIhat is: ', num2str(MIhat)]);






%% kmedoids
% for nCluster = [5,7],
%    disp(['Cluster number is: ', int2str(nCluster)]);
%    for file = (1:10),
%        filename=['MNIST/',int2str(nCluster),...
%            'Class/',int2str(file),'.mat'];
%        disp(['Begin testing file: ',filename]);      
%        load('MNIST/MNIST.mat');
%        load(filename);
%        fea = fea(sampleIdx,:);
%        gnd = gnd(sampleIdx,:);
%        fea(:,zeroIdx) = [];
%        tic;
%        [idx, C, sumD, D] = my_kmedoids(fea, nCluster);
%        toc;
%        idx = bestMap(gnd, idx);
%        accuracy = length(find(gnd == idx))/length(gnd);
%        disp(['The accuracy is: ', num2str(accuracy * 100), '%']);
%        MIhat = MutualInfo(gnd,idx);
%        disp(['The MIhat is: ', num2str(MIhat)]);
%    end
% end
% 
% disp('Cluster number is: 10');
% disp('Begin testing file: MNIST/MNIST.mat');
% load('MNIST/MNIST.mat');
% tic;
% [idx, C, sumD, D] = my_kmedoids(fea, 10);
% toc;
% idx = bestMap(gnd, idx);
% accuracy = length(find(gnd == idx))/length(gnd);
% disp(['The accuracy is: ', num2str(accuracy * 100), '%']);
% MIhat = MutualInfo(gnd,idx);
% disp(['The MIhat is: ', num2str(MIhat)]);
% 
% for nCluster = [5,10],
%    disp(['Cluster number is: ', int2str(nCluster)]);
%    for file = (1:10),
%        filename=['COIL20/',int2str(nCluster),...
%            'Class/',int2str(file),'.mat'];
%        disp(['Begin testing file: ',filename]);
%        load('COIL20/COIL20.mat');
%        load(filename);
%        fea = fea(sampleIdx,:);
%        gnd = gnd(sampleIdx,:);
%        fea(:,zeroIdx) = [];
%        tic;
%        [idx, C, sumD, D] = my_kmedoids(fea, nCluster);
%        toc;
%        idx = bestMap(gnd, idx);
%        accuracy = length(find(gnd == idx))/length(gnd);
%        disp(['The accuracy is: ', num2str(accuracy * 100), '%']);
%        MIhat = MutualInfo(gnd,idx);
%        disp(['The MIhat is: ', num2str(MIhat)]);
%    end
% end
% 
% disp('Cluster number is: 20');
% disp('Begin testing file: COIL20/COIL20.mat');
% load('COIL20/COIL20.mat');
% tic;
% [idx, C, sumD, D] = my_kmedoids(fea, 20);
% toc;
% idx = bestMap(gnd, idx);
% accuracy = length(find(gnd == idx))/length(gnd);
% disp(['The accuracy is: ', num2str(accuracy * 100), '%']);
% MIhat = MutualInfo(gnd,idx);
% disp(['The MIhat is: ', num2str(MIhat)]);





%% GMM
% for nCluster = [5,7],
%    disp(['Cluster number is: ', int2str(nCluster)]);
%    for file = (1:10),
%        filename=['MNIST/',int2str(nCluster),...
%            'Class/',int2str(file),'.mat'];
%        disp(['Begin testing file: ',filename]);      
%        load('MNIST/MNIST.mat');
%        load(filename);
%        fea = fea(sampleIdx,:);
%        gnd = gnd(sampleIdx,:);
%        fea(:,zeroIdx) = [];
%        tic;
%        idx = my_gmm(fea, nCluster);
%        toc;
%        idx = bestMap(gnd, idx);
%        accuracy = length(find(gnd == idx))/length(gnd);
%        disp(['The accuracy is: ', num2str(accuracy * 100), '%']);
%        MIhat = MutualInfo(gnd,idx);
%        disp(['The MIhat is: ', num2str(MIhat)]);
%    end
% end
% 
% disp('Cluster number is: 10');
% disp('Begin testing file: MNIST/MNIST.mat');
% load('MNIST/MNIST.mat');
% tic;
% idx = my_gmm(fea, 10);
% toc;
% idx = bestMap(gnd, idx);
% accuracy = length(find(gnd == idx))/length(gnd);
% disp(['The accuracy is: ', num2str(accuracy * 100), '%']);
% MIhat = MutualInfo(gnd,idx);
% disp(['The MIhat is: ', num2str(MIhat)]);

% for nCluster = [5,10],
%    disp(['Cluster number is: ', int2str(nCluster)]);
%    for file = (1:10),
%        filename=['COIL20/',int2str(nCluster),...
%            'Class/',int2str(file),'.mat'];
%        disp(['Begin testing file: ',filename]);
%        load('COIL20/COIL20.mat');
%        load(filename);
%        fea = fea(sampleIdx,:);
%        gnd = gnd(sampleIdx,:);
%        fea(:,zeroIdx) = [];
%        tic;
%        idx = my_gmm(fea, nCluster);
%        toc;
%        idx = bestMap(gnd, idx);
%        accuracy = length(find(gnd == idx))/length(gnd);
%        disp(['The accuracy is: ', num2str(accuracy * 100), '%']);
%        MIhat = MutualInfo(gnd,idx);
%        disp(['The MIhat is: ', num2str(MIhat)]);
%    end
% end
% 
% disp('Cluster number is: 20');
% disp('Begin testing file: COIL20/COIL20.mat');
% load('COIL20/COIL20.mat');
% tic;
% idx = my_gmm(fea, 20);
% toc;
% idx = bestMap(gnd, idx);
% accuracy = length(find(gnd == idx))/length(gnd);
% disp(['The accuracy is: ', num2str(accuracy * 100), '%']);
% MIhat = MutualInfo(gnd,idx);
% disp(['The MIhat is: ', num2str(MIhat)]);





%% Spectral Clustering
% for nCluster = [5,7],
%    disp(['Cluster number is: ', int2str(nCluster)]);
%    for file = (1:10),
%        filename=['MNIST/',int2str(nCluster),...
%            'Class/',int2str(file),'.mat'];
%        disp(['Begin testing file: ',filename]);      
%        load('MNIST/MNIST.mat');
%        load(filename);
%        fea = fea(sampleIdx,:);
%        gnd = gnd(sampleIdx,:);
%        fea(:,zeroIdx) = [];
%        
%        disp('  Unnormalized spectral clustering...')
%        tic;
%        idx = my_spectral(fea, nCluster, 0);
%        toc;
%        idx = bestMap(gnd, idx);
%        accuracy = length(find(gnd == idx))/length(gnd);
%        disp(['    The accuracy is: ', num2str(accuracy * 100), '%']);
%        MIhat = MutualInfo(gnd,idx);
%        disp(['    The MIhat is: ', num2str(MIhat)]);
%        
%        disp('  Normalized spectral clustering...')
%        tic;
%        idx = my_spectral(fea, nCluster, 1);
%        toc;
%        idx = bestMap(gnd, idx);
%        accuracy = length(find(gnd == idx))/length(gnd);
%        disp(['    The accuracy is: ', num2str(accuracy * 100), '%']);
%        MIhat = MutualInfo(gnd,idx);
%        disp(['    The MIhat is: ', num2str(MIhat)]);
%    end
% end
% 
% disp('Cluster number is: 10');
% disp('Begin testing file: MNIST/MNIST.mat');
% load('MNIST/MNIST.mat');
%        
% tic;
% idx = my_spectral(fea, 10, 0);
% toc;
% idx = bestMap(gnd, idx);
% accuracy = length(find(gnd == idx))/length(gnd);
% disp(['    The accuracy is: ', num2str(accuracy * 100), '%']);
% MIhat = MutualInfo(gnd,idx);
% disp(['    The MIhat is: ', num2str(MIhat)]);
% 
% disp('  Normalized spectral clustering...')
% tic;
% idx = my_spectral(fea, 10, 1);
% toc;
% idx = bestMap(gnd, idx);
% accuracy = length(find(gnd == idx))/length(gnd);
% disp(['    The accuracy is: ', num2str(accuracy * 100), '%']);
% MIhat = MutualInfo(gnd,idx);
% disp(['    The MIhat is: ', num2str(MIhat)]);
% 
% for nCluster = [5,10],
%    disp(['Cluster number is: ', int2str(nCluster)]);
%    for file = (1:10),
%        filename=['COIL20/',int2str(nCluster),...
%            'Class/',int2str(file),'.mat'];
%        disp(['Begin testing file: ',filename]);
%        load('COIL20/COIL20.mat');
%        load(filename);
%        fea = fea(sampleIdx,:);
%        gnd = gnd(sampleIdx,:);
%        fea(:,zeroIdx) = [];
%        
%        disp('  Unnormalized spectral clustering...')
%        tic;
%        idx = my_spectral(fea, nCluster, 0);
%        toc;
%        idx = bestMap(gnd, idx);
%        accuracy = length(find(gnd == idx))/length(gnd);
%        disp(['    The accuracy is: ', num2str(accuracy * 100), '%']);
%        MIhat = MutualInfo(gnd,idx);
%        disp(['    The MIhat is: ', num2str(MIhat)]);
%        
%        disp('  Normalized spectral clustering...')
%        tic;
%        idx = my_spectral(fea, nCluster, 1);
%        toc;
%        idx = bestMap(gnd, idx);
%        accuracy = length(find(gnd == idx))/length(gnd);
%        disp(['    The accuracy is: ', num2str(accuracy * 100), '%']);
%        MIhat = MutualInfo(gnd,idx);
%        disp(['    The MIhat is: ', num2str(MIhat)]);
%    end
% end
% 
% disp('Cluster number is: 20');
% disp('Begin testing file: COIL20/COIL20.mat');
% load('COIL20/COIL20.mat');
% disp('  Unnormalized spectral clustering...')
%        
% tic;
% idx = my_spectral(fea, 20, 0);
% toc;
% idx = bestMap(gnd, idx);
% accuracy = length(find(gnd == idx))/length(gnd);
% disp(['    The accuracy is: ', num2str(accuracy * 100), '%']);
% MIhat = MutualInfo(gnd,idx);
% disp(['    The MIhat is: ', num2str(MIhat)]);
% 
% disp('  Normalized spectral clustering...')
% tic;
% idx = my_spectral(fea, 20, 1);
% toc;
% idx = bestMap(gnd, idx);
% accuracy = length(find(gnd == idx))/length(gnd);
% disp(['    The accuracy is: ', num2str(accuracy * 100), '%']);
% MIhat = MutualInfo(gnd,idx);
% disp(['    The MIhat is: ', num2str(MIhat)]);