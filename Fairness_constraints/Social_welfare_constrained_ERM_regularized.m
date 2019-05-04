function Social_welfare_constrained_ERM_regularized(dataset_name, Alpha, Tau)
    % Load X, Y, G, F
    load(['Data/',dataset_name]);
    
    n = length(Y);
    k = size(X,2); % Number of features  

    % loss = zeros(length(Alpha),length(Tau)); 
    W_all = zeros(k,length(Tau)); 
    Y_predicted = zeros(n,length(Tau));
    
    
    n_folds = max(F);
    for fold=1:n_folds
        fold
        X_test = X(F==fold,:);
        Y_test = Y(F==fold,:);
        X_train = X(F~=fold,:);
        Y_train = Y(F~=fold,:);
        
        n_train = length(Y_train); % Number of train instances
        n_test = length(Y_test); % Number of test instances


        tauIndex = 0;
        for tau=Tau
            tauIndex = tauIndex+1
            
            cvx_begin quiet
            cvx_precision high         
            variable W(k)
                minimize( norm( X_train * W - Y_train, 2 ))
                 subject to
                 ones(1,n_train)*(X_train * W) >= n_train*tau
            cvx_end

            Y_hat = X_test * W;

            % loss(alphaIndex,tauIndex) = mean((Y_test - Y_hat).^2);
            W_all(:, tauIndex) = W_all(:, tauIndex) + W;
            Y_predicted(F==fold, tauIndex) = Y_hat;
        end
    end
    W_all = W_all ./n_folds;
    save(['Output/',dataset_name,'_Yhat.mat'],'Y','Tau', 'W_all');
end

