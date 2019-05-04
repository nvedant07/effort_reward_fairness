% Preprocessing the Crime and Communities data

load('student')
% Crime_original.mat does not include non-predictive attributes: communityname,
% state, countyCode, communityCode, and fold

% Removed target variables other than total number of nonviolent crimes per
% capita
X = student(:,1:14);
Y = student(:, 15);
G = student(:,2);


n = length(Y); % Number of training instances
k = size(X,2); % Number of features


% Homogenization
X = [X, ones(n,1)];


% 10-fold cross validation
n_folds=5;
p = randperm(n);
X = X(p, :);
Y = Y(p ,:);
G = G(p ,:);
F = ceil((1:n)./(n/n_folds));

save(['student',num2str(n_folds),'.mat'],'X', 'Y', 'G', 'F');