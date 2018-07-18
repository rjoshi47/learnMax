clear ; close all; clc

fprintf('Loading data ...\n');
data = load('ex1data2.txt');

featureSize = 8;
featureValue = 9;

X1 = data(:, 1:featureSize);
y1 = data(:, featureValue);

X = X1(1:500,:);
y = y1(1:500,:);
m = length(y);

fprintf('Normalizing Features ...\n');
[X mu sigma] = featureNormalize(X);
X = [ones(m, 1) X];

fprintf('Running gradient descent ...\n');
% Choose some alpha value
alpha = 0.01;
num_iters = 400;

% Init Theta and Run Gradient Descent 
theta = zeros(featureSize+1, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

%Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

figure;
for i = 502:700,
  test = X1(i,:);
  meanDiff = test' - mu';
  sigInv = sigma.^(-1);
  sigMul = meanDiff.*(sigInv');
  
  price = [ones(1) sigMul']*theta;
  hold on;
  plot(i,y1(i), 'r.', 'linewidth', 3);
  hold on;
  plot(i,price, 'b.','linewidth', 3);
end
% ============================================================

fprintf(['Predicted price of a 1604 sq-ft, 3 br house ' ...
         '(using gradient descent):\n $%f\n'], price);