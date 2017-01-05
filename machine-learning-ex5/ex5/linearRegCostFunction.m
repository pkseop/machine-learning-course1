function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% Regularized linear regression cost function
J = 1/(2*m) * (sum(((X * theta) - y).^ 2)) + ( lambda/(2*m) * sum(theta(2:end, 1) .^ 2 ) );


% Regularized linear regression grdient descent
hx = (X * theta) - y;
colLen = size(X,2);
for j = 1:colLen
	grad(j,1) = sum( hx .* X(:,j) ) / m;
	if j>1
		grad(j,1) += lambda/m*theta(j,1);
	endif
end



% =========================================================================

grad = grad(:);

end
