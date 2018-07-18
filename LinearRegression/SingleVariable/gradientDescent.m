function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    XVect = X*theta - y;
    XVectSum = sum(XVect);
    Xx = X(:,2);
    XVectxi = XVect.*Xx;
    XVectxiSum = sum(XVectxi);
    factor = alpha/m;
    theta = theta - factor*[XVectSum; XVectxiSum];  
    J_history(iter) = computeCost(X, y, theta);

end

end
