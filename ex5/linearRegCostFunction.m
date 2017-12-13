function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
theta_new=[0;theta(2:size(theta,1))];
J = (sum(((X*theta)-y).^2) + (lambda.*(sum(theta_new.^2))))./(2.*m);

grad(1)=(sum(sum((X*theta-y).*X(:,1))))./m;

for (x=2:size(theta))
grad(x)=(sum(sum(((X*theta)-y).*X(:,x))) + (lambda.*(theta(x))))./m;
end
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%












% =========================================================================

grad = grad(:);

end
