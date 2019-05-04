function Split( dataset_name, prcnt )
% Split the data into train and test sets
    load([dataset_name,'.mat']);
    
    n = length(Y);
    p = randperm(n);
    
    X_temp = X(p, :);
    Y_temp = Y(p ,:);
    G_temp = G(p ,:);
    
    n_test = ceil(n * prcnt);
    X_test = X_temp(1:n_test,:);
    Y_test = Y_temp(1:n_test,:);
    G_test = G_temp(1:n_test,:);
    
    X_train = X_temp(n_test+1:n,:);
    Y_train = Y_temp(n_test+1:n,:);
    G_train = G_temp(n_test+1:n,:);
    
    save([dataset_name,'Split.mat'], 'X_train', 'Y_train', 'G_train', 'X_test', 'Y_test', 'G_test');
end

