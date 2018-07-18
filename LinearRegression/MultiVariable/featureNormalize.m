function [X_norm, mu, sigma] = featureNormalize(X)

X_norm = X;
m = length(X_norm);
mu = zeros(1, size(X_norm, 2));
mu = mean(X_norm);

X_norm = X_norm - repmat(mu, m, 1);

sigma = zeros(1, size(X_norm, 2));
sigma = std(X_norm);
sigmaInv = sigma.^(-1);
sigmaEye = diag(sigmaInv);

%sigma2 = [sigmaInv(1,1) 0; 0 sigmaInv(1,2)]

X_norm = X_norm*sigmaEye;

end
