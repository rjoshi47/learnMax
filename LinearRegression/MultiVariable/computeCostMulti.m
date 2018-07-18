function J = computeCostMulti(X, y, theta)

m = length(y); % number of training examples
diff = X*theta - y;
diff = diff.^2;
J = sum(diff)*(1/(2*m));

end
