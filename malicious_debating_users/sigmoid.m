function g = sigmoid(z)
  
%SIGMOID Compute sigmoid function
%   J = SIGMOID(z) computes the sigmoid of z.

% Return the following variable correctly 
g = zeros(size(z));

g = 1.0 ./ (1.0 + exp(-z));

end
