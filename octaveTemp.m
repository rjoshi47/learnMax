X_norm = X;
m = length(X_norm);
mu = zeros(1, size(X_norm, 2));
sigma = zeros(1, size(X_norm, 2));
mu = mean(X_norm);
sigma = std(X_norm);
X_norm = X_norm - repmat(mu, m, 1);
sigma = sigma.^(-1);
sigma2 = [sigma(1,1) 0; 0 sigma(1,2)]
X_norm = X_norm*sigma2;



https://github.com/quinnliu/machineLearning/tree/master/supervisedLearning/linearRegressionInMultipleVariables
