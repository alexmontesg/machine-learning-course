function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

	m = length(y);
	[J, grad] = costFunction(theta, X, y);
	theta(1, 1) = 0;
	J += (lambda * sum((theta .^ 2)(:))) / (2 * m);
	prevG0 = grad(1);
	grad += (lambda * theta') / m;
	grad(1) = prevG0;

end
