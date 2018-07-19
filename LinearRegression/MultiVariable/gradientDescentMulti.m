function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)

m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    XVect = X*theta - y;
    dec = XVect'*X;
    factor = alpha/m;
    theta = theta - factor*(dec');  
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
