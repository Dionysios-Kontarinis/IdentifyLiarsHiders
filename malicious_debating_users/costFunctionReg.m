function [J, grad] = costFunctionReg(theta, X, y, lambda)
  
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% To be returned correctly
J = 0;
grad = zeros(size(theta));

J = (1/m) * sum( -y .* log(sigmoid(X*theta)) - (1-y) .* log(1 - sigmoid(X*theta)) ) + ... 
       lambda / (2 * m) * sum(theta(2:end).^2); % Don't consider theta(1) in the last sum.

% Attention: grad(1) differs from the other elements of the 'grad' vector, as follows: 
grad(1) = (1/m) * ( (sigmoid(X*theta)-y)' * X(:,1) );

grad(2:end) = (1/m) * ( (sigmoid(X*theta)-y)' * X(:,2:end) )' + lambda * (1/m) * theta(2:end);  

end
