function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

	m = length(y);
	hypothesis = sigmoid(X * theta);
	theta = [0 ; theta(2 : size(theta), :)];
	J = ((-y' * log(hypothesis) - (1 - y)' * log(1 - hypothesis)) / m) + (lambda * sum((theta .^ 2)(:))) / (2 * m);
	grad = (X' * (hypothesis - y) + lambda * theta) / m;
end
