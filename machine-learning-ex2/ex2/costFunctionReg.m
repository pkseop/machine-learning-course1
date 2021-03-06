function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% get sigmoid values
z = X * theta;
hx = sigmoid(z);
% cost function
J = ( sum( (-y .* log(hx)) - ((1-y) .* log(1-hx)) ) / m ) + ( lambda/(2*m) * sum(theta(2:end, 1) .^ 2 ) );
% grdient descent
colLen = size(X,2);
for i = 1:colLen
	grad(i,1) = sum( (hx - y) .* X(:,i) ) / m;
	if i>1
		grad(i,1) += lambda/m*theta(i,1);
	endif
end




% =============================================================

end
